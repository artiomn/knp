/**
 * @file message_bus.cu
 * @brief Message bus implementation.
 * @kaspersky_support Artiom N.
 * @date 21.02.2025
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>
#include <vector>
#include <boost/mp11/algorithm.hpp>
#include <boost/preprocessor.hpp>
#include <cuda/std/detail/libcxx/include/algorithm>
#include <thrust/device_ptr.h>
#include <knp/meta/macro.h>
#include "message_bus.cuh"

#include "../cuda_lib/vector.cuh"
#include "../cuda_lib/get_blocks_config.cuh"

// Как у нас работает бэк:
// 0. Мы индексируем index_messages() сообщения нужных нам типов (или всех типов?) и сохраняем вектор индексов.
// 1. каждая популяция получает входные сообщения и формирует, но не отправляет выходное сообщение.
//  1.1. Входные сообщения выдаются в виде "указатель на все сообщения + индексы интересующих"
// 2. когда все популяции отработали, мы:
//  2.1. чистим шину clean()
//  2.2. получаем сообщения receive_message() от популяций (в цикле)
//  2.3. отправляем сообщения в эндпойнт и получаем сообщения из эндпойнта (synchronize)
// потом мы проходим по проекциям (параллельно или нет), и они формируют сообщения
// мы чистим шину от спайковых сообщений (clean), получаем сообщения от проекций (в цикле) и отправляем их в эндпойнт.
// таким образом, конкретного step-а у нас попросту не образуется. Степ состоит из clear(),
// for(...) if(get_num_messages > 0) send_message(get_stored_messages) и sync().
// Все эти функции вызываются из бэкенда чем-то вроде do_message_exchange().


namespace knp::backends::gpu::cuda
{
template <class T>
using DevVec = device_lib::CUDAVector<T>;

/**
 * @brief Find messages with a given sender.
 * @param senders vector of senders, we select one based on thread
 */
//__global__ void find_by_sender(
//    const thrust::device_vector<cuda::UID> &senders,
//    const CUDAMessageBus::MessageBuffer &messages,
//    thrust::device_ptr<DevVec<uint64_t>> sub_message_indices,
//    int type_index)
//{
//    int sender_index = blockIdx.x + threadIdx.x;
//    if (sender_index >= senders.size()) return;
//    cuda::UID uid = senders[sender_index];
//    for (uint64_t i = 0; i < messages.size(); ++i)
//    {
//        const cuda::MessageVariant &msg = messages[i];
//        if (msg.index() != type_index) continue;
//        cuda::UID msg_uid = ::cuda::std::visit([](const auto &msg) { return msg.header_.sender_uid_; }, msg);
//        // if (msg_uid == uid) sub_message_indices[sender_index].push_back(msg_uid);
////        if (msg_uid == uid) (sub_message_indices + sender_index)->push_back(i);
//    }
//}


__global__ void find_subscription_by_receiver(const Subscription *subscriptions, size_t size, const UID receiver,
                                              size_t type, size_t *index_out)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    const Subscription &sub = subscriptions[i];
    if (sub.type() != type) return;
    if (sub.get_receiver_uid() == receiver) *index_out = i; // Should only work once, so no race condition problems.
}


template <typename MessageType>
__host__ size_t CUDAMessageBus::find_subscription(const cuda::UID &receiver)
{
    if (!subscriptions_.size()) return 0;
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(subscriptions_.size());

    size_t *index_out;
    cudaMalloc(&index_out, sizeof(size_t));
    size_t subscriptions_size = subscriptions_.size();
    cudaMemcpy(index_out, &subscriptions_size, sizeof(size_t), cudaMemcpyHostToDevice);
    constexpr size_t type_index = boost::mp11::mp_find<MessageVariant, MessageType>::value;

    find_subscription_by_receiver<<<num_blocks, num_threads>>>(subscriptions_.data(), subscriptions_.size(), receiver,
                                                               type_index, index_out);

    size_t result;
    cudaMemcpy(&result, index_out, sizeof(size_t), cudaMemcpyDeviceToHost);
    cudaFree(index_out);
    return result;
}


template <typename MessageType>
__host__ bool CUDAMessageBus::unsubscribe(const cuda::UID &receiver)
{
    SPDLOG_DEBUG("Unsubscribing");
    size_t sub_index = find_subscription<MessageType>(receiver);
    if (sub_index >= subscriptions_.size())
    {
        SPDLOG_TRACE("No subscriptions found to unsubscribe from: returned {}", sub_index);
        return false;
    }
    SPDLOG_TRACE("Removing subscription #{}", sub_index);
    subscriptions_.erase(subscriptions_.begin() + sub_index, subscriptions_.begin() + sub_index + 1);
    SPDLOG_TRACE("Done unsubscribing");
    return true;
}


__host__ void CUDAMessageBus::remove_receiver(const cuda::UID &receiver)
{
    for (auto sub_iter = subscriptions_.begin(); sub_iter != subscriptions_.end(); ++sub_iter)
    {
        // TODO: Finish
    }
}


// This is not threadsafe, make sure it's not run in parallel.
__host__ __device__ void CUDAMessageBus::send_message(const cuda::MessageVariant &message)
{
    messages_to_route_.push_back(message);
}


// first we collect all messages and put it into common buffer. Then for each subscription we check if it is their
// message, this is index_messages(). Then we extract a message by receiver. Or do we need it? What if extracting is
// done by asking a subscription and using a kernel? Then we don't need a subscription as a template, but it just has
// a type index. We say "receiver" and "type", the Bus finds a subscription by a kernel, and uses subscription to find
// messages by a second kernel, no indexing required. Yeah, let's do it like this. A subscription is never reused.


/**
 * @brief Find all message indices that correspond to the current subscription.
 * @param messages pointer to messages
 * @param messages_size number of messages
 * @param subscription the subscription used for searching
 * @param indexes
 * @param counter a zero-initialized counter, it would be equal to number of found messages after the function finishes
 * @return
 */
__global__ void find_messages_kernel(const MessageVariant *messages, size_t messages_size, Subscription *subscription,
                              uint64_t *indices, unsigned long long *counter)
{
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= messages_size) return;
    if (subscription->is_my_message(messages[i]))
    {
        uint64_t index = atomicAdd(counter, 1ull); // atomicAdd doesn't work with uint64_t but does with ULL.
        indices[index] = i;
    }
}


template <class MessageType>
device_lib::CUDAVector<uint64_t> CUDAMessageBus::unload_messages(const cuda::UID &receiver_uid)
{
    size_t sub_index = find_subscription<MessageType>(receiver_uid);
    if (sub_index >= subscriptions_.size()) return device_lib::CUDAVector<uint64_t>{};
    if (!messages_to_route_.size()) return device_lib::CUDAVector<uint64_t>{};

    Subscription subscription = subscriptions_.copy_at(sub_index);
    constexpr size_t message_type = boost::mp11::mp_find<MessageVariant, MessageType>();
    if (subscription.type() != message_type) return device_lib::CUDAVector<uint64_t>{};

    auto [num_blocks, num_threads] = device_lib::get_blocks_config(messages_to_route_.size());
    unsigned long long *counter;
    cudaMalloc(&counter, sizeof(uint64_t));
    uint64_t *indices;
    cudaMalloc(&indices, sizeof(uint64_t) * messages_to_route_.size());
    find_messages_kernel<<<num_blocks, num_threads>>>(messages_to_route_.data(), messages_to_route_.size(),
               subscriptions_.data() + sub_index, indices, counter);
    // Here we have a set of message indexes. Is that enough? I think it is.
    size_t cpu_counter;
    cudaMemcpy(&cpu_counter, counter, sizeof(cpu_counter), cudaMemcpyDeviceToHost);
    cudaFree(counter);
    device_lib::CUDAVector<uint64_t> result(cpu_counter);
    auto [num_blocks_copy, num_threads_copy] = device_lib::get_blocks_config(cpu_counter);
    device_lib::copy_kernel<<<num_blocks_copy, num_threads_copy>>>(result.data(), cpu_counter, indices);
    cudaFree(indices);
    return result;
}


__global__ void get_message_kernel(const MessageVariant *var, int *type, const void **msg) {
    int type_val = var->index();
    switch (type_val)
    {
        // TODO : Add more after adding new messages
        case 0:
            *msg = ::cuda::std::get_if<0>(var);
            break;
        case 1:
            *msg = ::cuda::std::get_if<1>(var);
            break;
        default:
            *msg = nullptr;
    }
    *type = type_val;
}


namespace cm = knp::backends::gpu::cuda;

template
__host__ bool cm::CUDAMessageBus::subscribe<SpikeMessage>(const cm::UID&, const std::vector<cuda::UID>&);

template
__host__ bool cm::CUDAMessageBus::subscribe<SynapticImpactMessage>(const cm::UID&, const std::vector<cuda::UID>&);

template
__host__ cm::device_lib::CUDAVector<uint64_t> cm::CUDAMessageBus::unload_messages<SpikeMessage>(
        const cm::UID &receiver_uid);

template
__host__ cm::device_lib::CUDAVector<uint64_t> cm::CUDAMessageBus::unload_messages<SynapticImpactMessage>(
        const cm::UID &receiver_uid);


#define INSTANCE_MESSAGES_FUNCTIONS(n, template_for_instance, message_type)                \
    template bool CUDAMessageBus::unsubscribe<cm::message_type>(const cuda::UID &receiver);

BOOST_PP_SEQ_FOR_EACH(INSTANCE_MESSAGES_FUNCTIONS, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_CUDA_MESSAGES))


}  // namespace knp::backends::gpu::cuda

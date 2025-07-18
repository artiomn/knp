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

#include <boost/mp11/algorithm.hpp>
#include <cuda/std/detail/libcxx/include/algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <knp/meta/macro.h>
#include "message_bus.cuh"



namespace knp::backends::gpu::cuda
{
constexpr int threads_per_block = 256;


template <class T>
using DevVec = thrust::device_vector<T>;


    template <typename MessageType>
__device__ bool CUDAMessageBus::subscribe(const UID &receiver, const thrust::device_vector<UID> &senders)
{
    for (const auto &subscr : subscriptions_)
    {
        const bool is_sub_exists = ::cuda::std::visit(
            [&receiver](auto &arg)
            {
                using T = std::decay_t<decltype(arg)>;
                return std::is_same<MessageType, typename T::MessageType>::value &&
                       (arg.get_receiver_uid() == receiver);
            },
            subscr);

        // TODO: Check, that senders contain all senders in the formal parameter or `senders` has something new?
        if (is_sub_exists)
        {
            return false;
        }
    }

    subscriptions_.push_back(Subscription<MessageType>(receiver, senders));

    return true;
}


// Допустим мы хотим сделать плоский вектор для индексов сообщений, потому что иначе дофига раз выделять и пушить.
// Тогда нам надо иметь вектор размера "сумма всех отправителей для каждого получателя * число сообщений". Это значит 
// что? Это значит что нам надо будет передавать вектор на начало своего сегмента -- а потом разбирать результат.
// Чтобы разобрать результат, нужно иметь вектор индексов начала подписок. Это поможет потом разобрать сообщения... 
// А размер? Там должен быть размер


// Нам надо на хосте иметь размеры подписок, чтобы выделить вектор. Но подписки у нас в cuda-variant.
// Так что обрабатываем их в куда-ядре
__global__ void get_subscription_size(const CUDAMessageBus::SubscriptionContainer &subscriptions, 
                                      thrust::device_vector<uint64_t> &result)
{
    uint64_t id = threadIdx.x + blockIdx.x;
    if (id >= subscriptions.size()) return;
    uint64_t senders_num = ::cuda::std::visit([](const auto &s) 
    { 
        return s.get_senders().size(); 
    }, static_cast<SubscriptionVariant>(subscriptions[id]));
    result[id] = senders_num;
}


__host__ thrust::device_vector<uint64_t> get_senders_numbers(const CUDAMessageBus::SubscriptionContainer &subscriptions)
{
    thrust::device_vector<uint64_t> result(subscriptions.size());
    uint64_t num_threads = std::min<uint64_t>(threads_per_block, subscriptions.size());
    uint64_t num_blocks = (subscriptions.size() - 1) / threads_per_block + 1;
    get_subscription_size<<<num_blocks, num_threads>>>(subscriptions, result);
    return result;
}


// фигачим вектор, его задача в том чтобы для каждой подписки и каждого отправителя был свой вектор, зарезервированный под размер
// всех сообщений. На самом деле бы лучше сделать это для каждой подписки нужного типа, но логика будет сложнее.
DevVec<DevVec<DevVec<uint64_t>>> reserve_vector(const CUDAMessageBus::SubscriptionContainer &subscriptions, uint64_t size_z)
{
    DevVec<DevVec<DevVec<uint64_t>>> res;
    uint64_t size_x = subscriptions.size();
    thrust::host_vector<uint64_t> senders_numbers = get_senders_numbers(subscriptions);
    res.reserve(size_x);
    for (uint64_t i = 0; i < size_x; ++i)
    {
        DevVec<DevVec<uint64_t>> buf;
        buf.reserve(senders_numbers[i]);
        for (uint64_t j = 0; j < senders_numbers[i]; ++j)
        {
            DevVec<uint64_t> sub_buf;
            sub_buf.reserve(size_z);
            buf.push_back(std::move(sub_buf));
        }
        res.push_back(std::move(buf));
    }
    return res;
}


__device__ int find_by_sender(
    const thrust::device_vector<cuda::UID> &senders, 
    const CUDAMessageBus::MessageBuffer &messages,
    DevVec<DevVec<uint64_t>> &sub_message_indices,
    int type_index)
{
    int sender_index = blockIdx.x + threadIdx.x;
    if (sender_index >= senders.size()) return;
    cuda::UID uid = senders[sender_index];
    for (uint64_t i = 0; i < messages.size(); ++i) 
    {
        const cuda::MessageVariant &msg = messages[i];
        if (msg.index() != type_index) continue;
        cuda::UID msg_uid = ::cuda::std::visit([](const auto &msg) {return msg.header_.sender_uid_; }, msg);
        // if (msg_uid == uid) sub_message_indices[sender_index].push_back(msg_uid); 
        thrust::device_ptr<DevVec<uint64_t>> ptr = sub_message_indices.data();
        if (msg_uid == uid) (ptr + sender_index)->push_back(msg_uid); 

    }
}


// Так, надо найти вектор сообщений с известным получателем и известного типа. Это несложно.
// Что нам надо: для каждого получателя получить отправителей. Запустить поиск по отправителю. Собрать результаты в вектор.
// Что нам надо для верхней функции: набор подписок и индекс типа, вектор сообщений, размеры для набора и вектора.
// Ещё нужен вектор для результата: 
__global__ void find_messages_by_receiver(
        const CUDAMessageBus::SubscriptionContainer &subscriptions,
        const CUDAMessageBus::MessageBuffer &messages,
        DevVec<DevVec<DevVec<uint64_t>>> &message_indices,
        int type_index)
{
    uint64_t sub_index = threadIdx.x + blockIdx.x;
    if (sub_index >= subscriptions.size()) return;
    const SubscriptionVariant &subscription = subscriptions[sub_index];
    if (subscription.index() != type_index) return;
    const DevVec<cuda::UID> &senders = ::cuda::std::visit([](auto &sub) { return sub.get_senders(); }, subscription);
    uint64_t buf_size = messages.size();

    // Find number of threads and blocks
    const int num_threads = std::min<int>(threads_per_block, subscriptions.size());
    const int num_blocks = subscriptions.size() / threads_per_block + 1;
    find_by_sender<<<num_blocks, num_threads>>>(senders, messages, message_indices[sub_index], type_index);
}


template<class MessageType>
__host__ thrust::device_vector<thrust::device_vector<thrust::device_vector<uint64_t>>> CUDAMessageBus::index_messages()
{
    uint64_t buf_size = messages_to_route_.size();
    // constexpr int type_index = boost::mp_find<MessageVariant, MessageType>::value;
    constexpr int type_index = 0; // TODO fix the code above.
    //Reserve memory:
    // Triple vector: receiver * senders * all_messages
    auto found_messages_indices = reserve_vector(subscriptions_, messages_to_route_.size());
    // Find number of threads and blocks and run the core.
    const int num_threads = std::min<int>(threads_per_block, subscriptions_.size());
    const int num_blocks = subscriptions_.size() / threads_per_block + 1;
    find_messages_by_receiver<<<num_blocks, num_threads>>>(subscriptions_, messages_to_route_, found_messages_indices, type_index);
    return found_messages_indices;
}




template <typename MessageType>
__device__ bool CUDAMessageBus::unsubscribe(const UID &receiver)
{
    auto sub_iter = thrust::find_if(thrust::device, subscriptions_.begin(), subscriptions_.end(),
    [&receiver](const cuda::SubscriptionVariant &subscr) -> bool
    {
        return std::visit([&receiver](const auto &arg)
        {
            using T = std::decay_t<decltype(arg)>;
            return std::is_same<MessageType, typename T::MessageType>::value && (arg.get_receiver_uid() == receiver);
        }, subscr);
    });

    if (subscriptions_.end() == sub_iter) return false;

    subscriptions_.erase(sub_iter);

    return true;
}


__device__ void CUDAMessageBus::remove_receiver(const UID &receiver)
{
    for (auto sub_iter = subscriptions_.begin(); sub_iter != subscriptions_.end(); ++sub_iter)
    {
/*        ::cuda::std::visit([&receiver](auto &&arg)
        {
            return arg.get_receiver_uid() == receiver;
        }, *sub_iter);
*/
    }

/*    if (subscriptions_.end() == sub_iter) return;

    subscriptions_.erase(sub_iter);*/
}


// This is not threadsafe, make sure it's not run in parallel.
__device__ void CUDAMessageBus::send_message(const cuda::MessageVariant &message)
{
    messages_to_route_.push_back(message);
}


__device__ size_t CUDAMessageBus::step()
{
    // Как у нас работает бэк: каждая популяция получает входные сообщения и формирует, но не отправляет выходное сообщение. 
    // когда все сообщения получены, мы чистим шину, получаем сообщения от популяций (в цикле) и отправляем их в эндпойнт.
    // потом мы проходим по проекциям (параллельно или нет), и они формируют сообщения
    // мы чистим шину от спайковых сообщений, получаем сообщения от проекций (в цикле) и отправляем их в эндпойнт.
    // таким образом, конкретного step-а у нас попросту не образуется. Степ состоит из clear(), 
    // for(...) if(get_num_messages > 0) send_message(get_stored_messages) и sync(). 
    // Все эти функции вызываются из бэкенда чем-то вроде do_message_exchange().
    return 1;
}


__device__ size_t CUDAMessageBus::route_messages()
{
    size_t count = 0;
    size_t num_messages = step();

    while (num_messages != 0)
    {
        count += num_messages;
        num_messages = step();
    }

    return count;
}


template <class MessageType>
__device__ void CUDAMessageBus::receive_messages(const cuda::UID &receiver_uid,
        thrust::device_vector<MessageType> &result_messages)
{
    // locate messages
    
}


}  // namespace knp::backends::gpu::cuda

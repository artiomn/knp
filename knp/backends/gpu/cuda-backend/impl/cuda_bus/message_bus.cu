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

#include <cuda/std/detail/libcxx/include/algorithm>
#include <knp/meta/macro.h>
#include "message_bus.cuh"


namespace knp::backends::gpu::cuda
{

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


__device__ void CUDAMessageBus::send_message(const cuda::MessageVariant &message)
{
}


__device__ size_t CUDAMessageBus::step()
{
/*    const std::lock_guard lock(mutex_);
    if (messages_to_route_.empty()) return 0;  // No more messages left for endpoints to receive.
    // Sending a message to every endpoint.
    auto message = std::move(messages_to_route_.back());
    // Remove message from container.
    messages_to_route_.pop_back();
    const knp::core::UID sender_uid = std::visit([](const auto &msg) { return msg.header_.sender_uid_; }, message);
    for (auto endpoint_data_containers : endpoint_data_)
    {
        auto recv_ptr = std::get<1>(endpoint_data_containers).lock();
        auto allowed_senders_ptr = std::get<2>(endpoint_data_containers).lock();
        // Skip all endpoints deleted after previous update(). They will be deleted at the next update().
        if ((!recv_ptr) || (!allowed_senders_ptr)) continue;
        if (allowed_senders_ptr->find(sender_uid) != allowed_senders_ptr->end())
        {
            recv_ptr->emplace_back(message);
        }
    }
*/
    /*const std::lock_guard lock(mutex_);
    if (messages_to_route_.empty()) return 0;  // No more messages left for endpoints to receive.
    // Sending a message to every endpoint.
    auto message = std::move(messages_to_route_.back());
    // Remove message from container.
    messages_to_route_.pop_back();
    const knp::core::UID sender_uid = std::visit([](const auto &msg) { return msg.header_.sender_uid_; }, message);
    for (auto endpoint_data_containers : endpoint_data_)
    {
        auto recv_ptr = std::get<1>(endpoint_data_containers).lock();
        auto allowed_senders_ptr = std::get<2>(endpoint_data_containers).lock();
        // Skip all endpoints deleted after previous update(). They will be deleted at the next update().
        if ((!recv_ptr) || (!allowed_senders_ptr)) continue;
        if (allowed_senders_ptr->find(sender_uid) != allowed_senders_ptr->end())
        {
            recv_ptr->emplace_back(message);
        }
    }*/

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


}  // namespace knp::backends::gpu::cuda

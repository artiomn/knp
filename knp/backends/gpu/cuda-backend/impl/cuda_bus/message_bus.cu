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

#include <knp/meta/macro.h>
#include "message_bus.cuh"

#include <zmq.hpp>


namespace knp::backends::gplu::impl
{
CUDAMessageBus::~CUDAMessageBus() = default;

CUDAMessageBus::CUDAMessageBus(CUDAMessageBus &&) noexcept = default;


template <typename MessageType>
_CCCL_DEVICE Subscription<MessageType> &CUDAMessageBus::subscribe(const UID &receiver, const std::vector<UID> &senders)
{
    for (const auto &subscr : subscriptions_)
    {
        if (subscr.get_receiver_uid() == receiver)
        {
            return;
        }
    }

    subscriptions_.push_back(Subscription<MessageType>(receiver, senders));
}


template <typename MessageType>
_CCCL_DEVICE bool CUDAMessageBus::unsubscribe(const UID &receiver)
{
}


_CCCL_DEVICE void CUDAMessageBus::remove_receiver(const UID &receiver)
{
}


_CCCL_DEVICE void CUDAMessageBus::send_message(const knp::core::messaging::MessageVariant &message)
{

}


template <class MessageType>
_CCCL_DEVICE std::vector<MessageType> CUDAMessageBus::unload_messages(const knp::core::UID &receiver_uid)
{
}



size_t CUDAMessageBus::step()
{
    const std::lock_guard lock(mutex_);
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

    return 1;
}


core::MessageEndpoint CUDAMessageBus::create_endpoint()
{
    const std::lock_guard lock(mutex_);

    using VT = std::vector<messaging::MessageVariant>;

    auto messages_to_send_v{std::make_shared<VT>()};
    auto recv_messages_v{std::make_shared<VT>()};

    auto endpoint = MessageEndpointCPU(std::make_shared<MessageEndpointCPUImpl>(messages_to_send_v, recv_messages_v));
    endpoint_data_.emplace_back(messages_to_send_v, recv_messages_v, endpoint.get_senders_ptr());
    return std::move(endpoint);
}


size_t CUDAMessageBus::step()
{
    const std::lock_guard lock(mutex_);
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

    return 1;
}


size_t CUDAMessageBus::route_messages()
{
    size_t count = 0;
    impl_->update();
    size_t num_messages = step();

    while (num_messages != 0)
    {
        count += num_messages;
        num_messages = step();
    }

    return count;
}


}  // namespace knp::backends::gpu::impl

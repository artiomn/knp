/**
 * @file message_bus.cuh
 * @brief CUDA message bus interface.
 * @kaspersky_support Artiom N.
 * @date 16.03.2025
 * @license Apache 2.0
 * @copyright © 2025 AO Kaspersky Lab
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

#pragma once

//#include <thrust/device_vector.h>

#include <knp/core/uid.h>
#include <knp/core/message_endpoint.h>
#include <cub/config.cuh>

#include <cuda/std/variant>

#include <functional>
#include <memory>
#include <utility>

#include <thrust/device_vector.h>

#include "subscription.cuh"
#include "../uid.cuh"
#include "messaging.cuh"
#include "../cuda_lib/vector.cuh"


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda
{
/**
 * @brief The MessageBus class is a definition of an interface to a message bus.
 */
class CUDAMessageBus
{
public:
    /**
     * @brief Construct GPU message bus.
     * @param external_endpoint message endpoint used for message exchange with host.
     */
    explicit CUDAMessageBus(knp::core::MessageEndpoint &&external_endpoint) :
        cpu_endpoint_{std::move(external_endpoint)}
    {}

public:
    /**
     * @brief Add a subscription to messages of the specified type from senders with given UIDs.
     * @note If the subscription for the specified receiver and message type already exists, the method updates the list
     * of senders in the subscription.
     * @tparam MessageType type of messages to which the receiver subscribes via the subscription.
     * @param receiver receiver UID.
     * @param senders vector of sender UIDs.
     * @return number of senders added to the subscription.
     */
    template <typename MessageType>
    __host__ bool subscribe(const UID &receiver, const thrust::device_vector<UID> &senders)
    {
        for (size_t index = 0; index < subscriptions_.size(); ++index)
        {
            const auto subscr = subscriptions_.copy_at(index);
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

    /**
     * @brief Unsubscribe from messages of a specified type.
     * @tparam MessageType type of messages to which the receiver is subscribed.
     * @param receiver receiver UID.
     * @return true if a subscription was deleted, false otherwise.
     */
    template <typename MessageType>
    __host__ bool unsubscribe(const UID &receiver) { return true; } // TODO TEMP

    /**
     * @brief Remove all subscriptions for a receiver with given UID.
     * @param receiver receiver UID.
     */
    __host__ void remove_receiver(const UID &receiver);

    /**
     * @brief Send a message to the message bus.
     * @param message message to send.
     */
    __device__ void send_message(const cuda::MessageVariant &message); // Done

    /**
     * @brief Delete all messages inside the bus.
     */
    __host__ void clear() { messages_to_route_.clear(); }

    /**
     * @brief Reserve bus buffer for messages.
     * @param num_messages number of messages.
     */
    __host__ void reserve_message_buffer(uint64_t num_messages) { messages_to_route_.reserve(num_messages); }

    /**
     * @brief Read messages of the specified type received via subscription.
     * @tparam MessageType type of messages to read.
     * @param receiver_uid receiver UID.
     * @return vector of messages.
     */
    template <class MessageType>
    __device__ void receive_messages(const cuda::UID &receiver_uid,
                                     device_lib::CUDAVector<MessageType> &result_messages);

    __device__ cuda::MessageVariant& get_message(uint64_t message_index);

public:
    /**
     * @brief Type of subscription container.
     */
    using SubscriptionContainer = device_lib::CUDAVector<SubscriptionVariant>;

    using MessageBuffer = device_lib::CUDAVector<cuda::MessageVariant>;

    /**
     * @brief Get a reference of the subscription container of the endpoint.
     * @return reference to the subscription container.
     */
    SubscriptionContainer& get_subscriptions() { return subscriptions_; }

private:
    /**
     * @brief Send messages to CPU endpoint.
     */
    __host__ int synchronize();

    /**
     *
     */
    template<class MessageType>
    __host__ thrust::device_vector<thrust::device_vector<thrust::device_vector<uint64_t>>> index_messages();

    /**
     * @brief Container that stores all the subscriptions for the current endpoint.
     */
    SubscriptionContainer subscriptions_;

    MessageBuffer messages_to_route_;

    knp::core::MessageEndpoint cpu_endpoint_;
};


template
__host__ bool CUDAMessageBus::subscribe<SpikeMessage>(const UID&, const thrust::device_vector<UID>&);

template
__host__ bool CUDAMessageBus::subscribe<SynapticImpactMessage>(const UID&, const thrust::device_vector<UID>&);

//template
//__host__ bool CUDAMessageBus::unsubscribe<SpikeMessage>(const UID &receiver);
//
//template
//__host__ bool CUDAMessageBus::unsubscribe<SynapticImpactMessage>(const UID &receiver);

}  // namespace knp::backends::gpu::cuda

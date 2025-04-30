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

#include <functional>
#include <memory>

#include <cuda/std/variant>
#include <thrust/device_vector.h>

#include <knp/core/uid.h>

#include <cub/config.cuh>
#include "subscription.cuh"
#include "cuda_common.cuh"
#include "messaging.cuh"


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
     * @brief Route some messages.
     * @return number of messages routed during the step.
     */
    _CCCL_DEVICE size_t step();

    /**
     * @brief Route messages.
     * @return number of messages routed.
     */
    _CCCL_DEVICE size_t route_messages();

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
    _CCCL_DEVICE bool subscribe(const UID &receiver, const thrust::device_vector<UID> &senders);

    /**
     * @brief Unsubscribe from messages of a specified type.
     * @tparam MessageType type of messages to which the receiver is subscribed.
     * @param receiver receiver UID.
     * @return true if a subscription was deleted, false otherwise.
     */
    template <typename MessageType>
    _CCCL_DEVICE bool unsubscribe(const UID &receiver);

    /**
     * @brief Remove all subscriptions for a receiver with given UID.
     * @param receiver receiver UID.
     */
    _CCCL_DEVICE void remove_receiver(const UID &receiver);

    /**
     * @brief Send a message to the message bus.
     * @param message message to send.
     */
    _CCCL_DEVICE void send_message(const cuda::MessageVariant &message);

    /**
     * @brief Read messages of the specified type received via subscription.
     * @note After reading the messages, the method clears them from the subscription.
     * @tparam MessageType type of messages to read.
     * @param receiver_uid receiver UID.
     * @return vector of messages.
     */
    template <class MessageType>
    _CCCL_DEVICE std::vector<MessageType> unload_messages(const UID &receiver_uid);

public:
    /**
     * @brief Type of subscription container.
     */
    using SubscriptionContainer = thrust::device_vector<SubscriptionVariant>;

    /**
     * @brief Get access to subscription container of the endpoint.
     * @return Reference to subscription container.
     */
    const SubscriptionContainer &get_subscriptions() const { return subscriptions_; }

private:
    /**
     * @brief Container that stores all the subscriptions for the current endpoint.
     */
    SubscriptionContainer subscriptions_;

    thrust::device_vector<cuda::MessageVariant> messages_to_route_;

    std::mutex mutex_;
};

}  // namespace knp::backends::gpu::impl

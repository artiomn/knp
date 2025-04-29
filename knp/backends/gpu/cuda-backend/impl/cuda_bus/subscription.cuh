/**
 * @file subscription.cuh
 * @brief Subscription class that determines message exchange between entities in the network.
 * @kaspersky_support
 * @date 15.03.2023
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

#pragma once

#include <cuda/std/back_insert_iterator.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/find.h>

#include "cuda_common.cuh"


namespace knp::backends::gpu::cuda
{

/**
 * @brief The Subscription class is used for message exchange between the network entities.
 * @tparam MessageT type of messages that are exchanged via the subscription.
 */
template <class MessageT>
class Subscription final
{
public:
    /**
     * @brief Message type.
     */
    using MessageType = MessageT;
    /**
     * @brief Internal container for messages of the specified message type.
     */
    using MessageContainerType = thrust::device_vector<MessageType>;

    /**
     * @brief Internal container for UIDs.
     */
    using UidSet = thrust::device_vector<UID>;
    // Subscription(const Subscription &) = delete;

public:
    /**
     * @brief Subscription constructor.
     * @param receiver receiver UID.
     * @param senders list of sender UIDs.
     */
    _CCCL_DEVICE Subscription(const UID &receiver, const thrust::device_vector<UID> &senders) : receiver_(receiver) { add_senders(senders); }

    /**
     * @brief Get list of sender UIDs.
     * @return senders UIDs.
     */
    [[nodiscard]] _CCCL_DEVICE const UidSet &get_senders() const { return senders_; }

    /**
     * @brief Get UID of the entity that receives messages via the subscription.
     * @return UID.
     */
    [[nodiscard]] _CCCL_DEVICE UID get_receiver_uid() const { return receiver_; }

    /**
     * @brief Unsubscribe from a sender.
     * @details If a sender is not associated with the subscription, the method doesn't do anything.
     * @param uid sender UID.
     * @return true if sender was deleted from subscription.
     */
    _CCCL_DEVICE bool remove_sender(const UID &uid)
    {
        auto erase_iter = thrust::find(thrust::device, senders_.begin(), senders_.end(), uid);
        if (senders_.end() == erase_iter) return false;
        senders_.erase(erase_iter);
        return true;
    }

    /**
     * @brief Add a sender with the given UID to the subscription.
     * @details If a sender is already associated with the subscription, the method doesn't do anything.
     * @param uid UID of the new sender.
     * @return true if sender added.
     */
    _CCCL_DEVICE bool add_sender(const UID &uid)
    {
        if (has_sender(uid)) return false;

        senders_.push_back(uid);

        return true;
    }

    /**
     * @brief Add several senders to the subscription.
     * @param senders vector of sender UIDs.
     * @return number of senders added.
     */
    _CCCL_DEVICE size_t add_senders(const thrust::device_vector<UID> &senders)
    {
        const size_t size_before = senders_.size();

        senders_.reserve(size_before + senders.size());
        thrust::copy(thrust::device, senders.begin(), senders.end(), cuda::back_inserter(senders_));

        return senders_.size() - size_before;
    }

    /**
     * @brief Check if a sender with the given UID exists.
     * @param uid sender UID.
     * @return `true` if the sender with the given UID exists, `false` if the sender with the given UID doesn't exist.
     */
    [[nodiscard]] _CCCL_DEVICE bool has_sender(const UID &uid) const
    {
        return thrust::find(thrust::device, senders_.begin(), senders_.end(), uid) != senders_.end();
    }

public:
    /**
     * @brief Add a message to the subscription.
     * @param message message to add.
     */
    _CCCL_DEVICE void add_message(MessageType &&message) { messages_.push_back(message); }
    /**
     * @brief Add a message to the subscription.
     * @param message constant message to add.
     */
    _CCCL_DEVICE void add_message(const MessageType &message) { messages_.push_back(message); }

    /**
     * @brief Get all messages.
     * @return reference to message container.
     */
    _CCCL_DEVICE MessageContainerType &get_messages() { return messages_; }
    /**
     * @brief Get all messages.
     * @return constant reference to message container.
     */
    _CCCL_DEVICE const MessageContainerType &get_messages() const { return messages_; }

    /**
     * @brief Remove all stored messages.
     */
    _CCCL_DEVICE void clear_messages() { messages_.clear(); }

private:
    /**
     * @brief Receiver UID.
     */
    const UID receiver_;

    /**
     * @brief Set of sender UIDs.
     */
    UidSet senders_;
    /**
     * @brief Message storage.
     */
    MessageContainerType messages_;
};

}  // namespace knp::core

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

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/find.h>
#include <thrust/execution_policy.h>

#include <boost/mp11.hpp>
#include <cuda/std/iterator>
#include <algorithm>

#include "cuda_common.cuh"
#include "messaging.cuh"


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
    __device__ Subscription(const UID &receiver, const thrust::device_vector<UID> &senders) :
        receiver_(receiver) { add_senders(senders); }

    /**
     * @brief Get list of sender UIDs.
     * @return senders UIDs.
     */
    [[nodiscard]] __device__ const UidSet &get_senders() const { return senders_; }

    /**
     * @brief Get UID of the entity that receives messages via the subscription.
     * @return UID.
     */
    [[nodiscard]] __device__ UID get_receiver_uid() const { return receiver_; }

    /**
     * @brief Unsubscribe from a sender.
     * @details If a sender is not associated with the subscription, the method doesn't do anything.
     * @param uid sender UID.
     * @return true if sender was deleted from subscription.
     */
    __device__ bool remove_sender(const UID &uid)
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
    __device__ bool add_sender(const UID &uid)
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
    __device__ size_t add_senders(const thrust::device_vector<UID> &senders)
    {
        const size_t size_before = senders_.size();

        senders_.reserve(size_before + senders.size());
        thrust::copy(thrust::device, senders.begin(), senders.end(), ::cuda::std::back_inserter(senders_));

        return senders_.size() - size_before;
    }

    /**
     * @brief Check if a sender with the given UID exists.
     * @param uid sender UID.
     * @return `true` if the sender with the given UID exists, `false` if the sender with the given UID doesn't exist.
     */
    [[nodiscard]] __device__ bool has_sender(const UID &uid) const
    {
        return thrust::find(thrust::device, senders_.begin(), senders_.end(), uid) != senders_.end();
    }

public:
    /**
     * @brief Add a message to the subscription.
     * @param message message to add.
     */
    __device__ void add_message(MessageType &&message) { messages_.push_back(message); }
    /**
     * @brief Add a message to the subscription.
     * @param message constant message to add.
     */
    __device__ void add_message(const MessageType &message) { messages_.push_back(message); }

    /**
     * @brief Get all messages.
     * @return reference to message container.
     */
    __device__ MessageContainerType &get_messages() { return messages_; }
    /**
     * @brief Get all messages.
     * @return constant reference to message container.
     */
    __device__ const MessageContainerType &get_messages() const { return messages_; }

    /**
     * @brief Remove all stored messages.
     */
    __device__ void clear_messages() { messages_.clear(); }

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


/**
 * @brief List of subscription types based on message types specified in `messaging::AllMessages`.
 */
using AllSubscriptions = boost::mp11::mp_transform<Subscription, cuda::AllMessages>;

/**
 * @brief Subscription variant that contains any subscription type specified in `AllSubscriptions`.
 * @details `SubscriptionVariant` takes the value of `std::variant<SubscriptionType_1,..., SubscriptionType_n>`,
 * where `SubscriptionType_[1..n]` is the subscription type specified in `AllSubscriptions`. \n For example, if
 * `AllSubscriptions` contains SpikeMessage and SynapticImpactMessage types, then `SubscriptionVariant =
 * std::variant<SpikeMessage, SynapticImpactMessage>`. \n `SubscriptionVariant` retains the same order of message
 * types as defined in `AllSubscriptions`.
 * @see ALL_MESSAGES.
 */
using SubscriptionVariant = boost::mp11::mp_rename<AllSubscriptions, ::cuda::std::variant>;


} // namespace knp::backends::gpu::cuda

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

// #include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>
// #include <thrust/find.h>
#include <thrust/execution_policy.h>

#include <boost/mp11.hpp>
#include <cuda/std/iterator>
#include <algorithm>

#include "../cuda_lib/vector.cuh"
#include "cuda_common.cuh"
#include "messaging.cuh"


namespace knp::backends::gpu::cuda
{

__global__ void has_sender_core(const UID &uid, device_lib::CudaVector<UID> senders, thrust::device_vector<bool> &results) 
{
    uint64_t index = threadIdx.x + blockIdx.x + blockDim.x;
    if (index >= senders.size()) return;
    results[index] = (senders[index] == uid);
}

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
    using UidSet = device_lib::CudaVector<UID>;
    // __host__ __device__ Subscription() : receiver_(to_gpu_uid(knp::core::UID{false})) {}
    __host__ __device__ Subscription() = default;
    __host__ __device__ Subscription(const Subscription &) = default;
    __host__ __device__ ~Subscription() = default;

public:
    /**
     * @brief Subscription constructor.
     * @param receiver receiver UID.
     * @param senders list of sender UIDs.
     */
    __host__ __device__ Subscription(const UID &receiver, const thrust::device_vector<UID> &senders) :
        receiver_(receiver) 
    { 
        #ifdef __CUDA_ARCH__
        for (size_t i = 0; i < senders.size(); ++i) add_sender(senders[i]);
        #else
        thrust::host_vector<UID> host_vec = senders;
        for (size_t i = 0; i < host_vec.size(); ++i) add_sender(host_vec[i]);
        #endif
    }

    /**
     * @brief Get list of sender UIDs.
     * @return senders UIDs.
     */
    [[nodiscard]] __device__ __host__ const UidSet &get_senders() const { return senders_; }

    // /**
    //  * @brief Get UID of the entity that receives messages via the subscription.
    //  * @return UID.
    //  */
    // [[nodiscard]] __device__ __host__ UID get_receiver_uid() const { return receiver_; }

    /**
     * @brief Unsubscribe from a sender.
     * @details If a sender is not associated with the subscription, the method doesn't do anything.
     * @param uid sender UID.
     * @return true if sender was deleted from subscription.
     */
    __device__ __host__ bool remove_sender(const UID &uid)
    {
        // auto erase_iter = thrust::find(thrust::device, senders_.begin(), senders_.end(), uid);
        uint64_t index = 0;
        for (uint64_t index = 0; index < senders_.size(); ++index)
        {
            if (senders_[index] == uid)
            {
                senders_.erase(index);
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Add a sender with the given UID to the subscription.
     * @details If a sender is already associated with the subscription, the method doesn't do anything.
     * @param uid UID of the new sender.
     * @return true if sender added.
     */
    __host__ __device__ bool add_sender(const UID &uid)
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
    __host__ __device__ size_t add_senders(const thrust::device_vector<UID> &senders)
    {
        size_t result = 0;
        for (size_t i = 0; i < senders.size(); ++i)
            result += add_sender(senders[i]);
        return result;
    }

    /**
     * @brief Check if a sender with the given UID exists.
     * @param uid sender UID.
     * @return `true` if the sender with the given UID exists, `false` if the sender with the given UID doesn't exist.
     */
    [[nodiscard]] __host__ __device__ bool has_sender(const UID &uid) const
    {
        #ifdef __CUDA_ARCH__
        for (size_t i = 0; i < senders_.size(); ++i)
            if (senders_[i] == uid) return true;
        return false;
        #else
        thrust::device_vector<bool> results(senders_.size(), false);
        size_t num_threads = std::min<size_t>(senders_.size(), 256); // TODO change 256 to named constant
        size_t num_blocks = (senders_.size() - 1) / num_threads + 1;
        has_sender_core<<<num_blocks, num_threads>>>(uid, senders_,results);
        return thrust::any_of(results.begin(), results.end(), ::cuda::std::identity{});
        #endif

    //     // return std::find(senders_.begin(), senders_.end(), uid) != senders_.end();
    //     // return thrust::find(thrust::host, senders_.begin(), senders_.end(), uid) != senders_.end();
    }

public:
    /**
     * @brief Add a message to the subscription.
     * @param message message to add.
     */
    // __device__ __host__ void add_message(MessageType &&message) { messages_.push_back(message); }
    /**
     * @brief Add a message to the subscription.
     * @param message constant message to add.
     */
    // __device__ void add_message(const MessageType &message) { messages_.push_back(message); }

    /**
     * @brief Get all messages.
     * @return reference to message container.
     */
    // __device__ MessageContainerType &get_messages() { return messages_; }
    /**
     * @brief Get all messages.
     * @return constant reference to message container.
     */
    // __device__ const MessageContainerType &get_messages() const { return messages_; }

    /**
     * @brief Remove all stored messages.
     */
    // __device__ void clear_messages() { messages_.clear(); }

private:
    /**
     * @brief Receiver UID.
     */
    UID receiver_;

    /**
     * @brief Set of sender UIDs.
     */
    UidSet senders_;
    /**
     * @brief Message storage.
     */
    // MessageContainerType messages_;
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

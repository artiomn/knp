/**
 * @file message_endpoint.h
 * @brief Message endpoint interface.
 * @kaspersky_support Artiom N.
 * @date 23.01.2025
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

#include <knp/core/messaging/message_envelope.h>
#include <knp/core/messaging/messaging.h>

#include <variant>
#include <vector>

#include <boost/mp11.hpp>
#include <boost/noncopyable.hpp>



/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::impl
{
/**
 * @brief The MessageEndpoint class is a definition of message endpoints.
 * @details You can use message endpoints to receive or send messages.
 */
class MessageEndpoint : private boost::noncopyable
{
public:
    /**
     * @brief List of subscription types based on message types specified in `messaging::AllMessages`.
     */
    using AllSubscriptions = boost::mp11::mp_transform<Subscription, messaging::AllMessages>;

    /**
     * @brief Subscription variant that contains any subscription type specified in `AllSubscriptions`.
     * @details `SubscriptionVariant` takes the value of `std::variant<SubscriptionType_1,..., SubscriptionType_n>`,
     * where `SubscriptionType_[1..n]` is the subscription type specified in `AllSubscriptions`. \n For example, if
     * `AllSubscriptions` contains SpikeMessage and SynapticImpactMessage types, then `SubscriptionVariant =
     * std::variant<SpikeMessage, SynapticImpactMessage>`. \n `SubscriptionVariant` retains the same order of message
     * types as defined in `AllSubscriptions`.
     * @see ALL_MESSAGES.
     */
    using SubscriptionVariant = boost::mp11::mp_rename<AllSubscriptions, cuda::std::variant>;

public:
    /**
     * @brief Get receiver UID from a subscription variant.
     * @param subscription subscription variant.
     * @return receiver UID.
     */
    static UID get_receiver_uid(const SubscriptionVariant &subscription);
    /**
     * @brief Get subscription key from a subscription variant.
     * @param subscription subscription variant.
     * @return pair of subscription index and subscription key.
     */
    static std::pair<size_t, UID> get_subscription_key(const SubscriptionVariant &subscription);

    /**
     * @brief Find index of an entity type in its variant.
     * @details For example, you can use the method to find an index of a message type in a message variant or an index
     * of a subscription type in a subscription variant.
     * @tparam Variant variant of one or more entity types.
     * @tparam Type entity type to search.
     */
    template <typename Variant, typename Type>
    static constexpr size_t get_type_index = boost::mp11::mp_find<Variant, Type>::value;

public:
    /**
     * @brief Move constructor for message endpoints.
     * @param endpoint endpoint to move.
     */
    MessageEndpoint(MessageEndpoint &&endpoint) noexcept;

    /**
     * @brief Avoid copy assignment of an endpoint.
     */
    MessageEndpoint &operator=(const MessageEndpoint &) = delete;

    /**
     * @brief Message endpoint destructor.
     */
    virtual ~MessageEndpoint();


protected:
    /**
     * @brief Message endpoint implementation.
     */
    std::shared_ptr<messaging::impl::MessageEndpointImpl> impl_;

protected:
    /**
     * @brief Message endpoint default constructor.
     */
    MessageEndpoint() = default;

};

}  // namespace knp::backends::gpu::impl

/**
 * @file message_bus_impl.h
 * @brief Message bus implementation interface class.
 * @kaspersky_support Vartenkov A.
 * @date 19.09.2023
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
#include <knp/core/message_endpoint.h>

/**
 * @brief Namespace for implementations of message bus.
 */
namespace knp::core::messaging::impl
{
/**
 * @brief Base class for different message bus implementations.
 */
class MessageBusImpl
{
public:
    /**
     * @brief Default virtual destructor.
     */
    virtual ~MessageBusImpl() = default;

    /**
     * @brief Send a message between endpoints.
     * @return number of messages routed during this step.
     */
    virtual size_t step() = 0;

    /**
     * @brief Create an endpoint that can be used for message exchange.
     * @return new endpoint.
     */
    [[nodiscard]] virtual MessageEndpoint create_endpoint() = 0;

    /**
     * @brief Update if needed. The function is to to be called once before message routing.
     */
    virtual void update() {}
};
}  // namespace knp::core::messaging::impl

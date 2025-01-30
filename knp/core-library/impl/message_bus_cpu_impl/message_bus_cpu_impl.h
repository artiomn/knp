/**
 * @file message_bus_cpu_impl.h
 * @brief CPU-based message bus implementation header.
 * @kaspersky_support Vartenkov A.
 * @date 18.09.2023
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

#include <knp/core/message_bus.h>

#include <message_bus_impl.h>

#include <list>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>


/**
 * @brief Namespace for implementations of message bus.
 */
namespace knp::core::messaging::impl
{
class MessageEndpointCPU;

class MessageBusCPUImpl : public MessageBusImpl
{
public:
    void update() override;
    size_t step() override;
    [[nodiscard]] core::MessageEndpoint create_endpoint() override;

private:
    // cppcheck-suppress unusedStructMember
    std::vector<knp::core::messaging::MessageVariant> messages_to_route_;
    std::list<std::tuple<
        std::weak_ptr<std::vector<messaging::MessageVariant>>, std::weak_ptr<std::vector<messaging::MessageVariant>>>>
        // cppcheck-suppress unusedStructMember
        endpoint_messages_;
    std::mutex mutex_;
};
}  // namespace knp::core::messaging::impl

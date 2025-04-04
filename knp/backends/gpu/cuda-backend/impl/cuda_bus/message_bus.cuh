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

#include <cub/config.cuh>

#include <functional>
#include <memory>

#include <cuda/std/variant>

#include "message_endpoint.cuh"


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::impl
{
/**
 * @brief The MessageBus class is a definition of an interface to a message bus.
 */
class CUDAMessageBus
{
public:
    /**
     * @brief Create a message bus with default implementation.
     * @return message bus.
     */
    static CUDAMessageBus _CCCL_DEVICE construct_bus() { return CUDAMessageBus(); }

    /**
     * @brief Default message bus constructor is deleted.
     * @note Use one of the static functions above.
     */
    _CCCL_DEVICE CUDAMessageBus() = delete;

    /**
     * @brief Move constructor.
     */
    _CCCL_DEVICE CUDAMessageBus(CUDAMessageBus &&) noexcept;

    /**
     * @brief Message bus destructor.
     */
    _CCCL_DEVICE ~CUDAMessageBus();

public:
    /**
     * @brief Create a new endpoint that sends and receives messages through the message bus.
     * @return new endpoint.
     * @see MessageEndpoint.
     */
    [[nodiscard]] _CCCL_DEVICE MessageEndpoint create_endpoint();

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

private:
    thrust::device_vector<knp::core::messaging::MessageVariant> messages_to_route_;

    // This is a list of endpoint data:
    // First vector is a pointer to a set of messages the endpoint is sending.
    // Second vector is a set of messages endpoint is receiving.
    // Third set contains message senders, and is kept updated by endpoints when they add or remove them.
    ::list<std::tuple<
        std::weak_ptr<std::vector<messaging::MessageVariant>>, std::weak_ptr<std::vector<messaging::MessageVariant>>,
        std::weak_ptr<std::unordered_set<knp::core::UID, knp::core::uid_hash>>>>
        endpoint_data_;
    std::mutex mutex_;
};

}  // namespace knp::backends::gpu::impl

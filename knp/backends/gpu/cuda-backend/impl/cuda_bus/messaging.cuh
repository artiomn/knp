/**
 * @file messages.h
 * @brief Messages file for CUDA.
 * @kaspersky_support Artiom N.
 * @date 20.04.2025
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

#include <boost/mp11.hpp>
#include <cuda/std/variant>

#include "message_header.cuh"
#include "spike_message.cuh"
#include "synaptic_impact_message.cuh"
#include "../cuda_lib/extraction.cuh"
#include "../cuda_lib/safe_call.cuh"


/**
 * @brief CUDA messaging namespace.
 */
namespace knp::backends::gpu::cuda
{
#define ALL_CUDA_MESSAGES SpikeMessage, SynapticImpactMessage

using AllCudaMessages = boost::mp11::mp_list<ALL_CUDA_MESSAGES>;

using MessageVariant = boost::mp11::mp_rename<AllCudaMessages, ::cuda::std::variant>;


__global__ void get_message_kernel(const MessageVariant *var, int *type, const void **msg);


template<>
MessageVariant extract<MessageVariant>(const MessageVariant *message);
}  // namespace knp::backends::gpu::cuda

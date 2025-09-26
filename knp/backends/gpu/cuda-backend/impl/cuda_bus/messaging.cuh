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
namespace knp::backends::gpu::cuda {
#define ALL_MESSAGES SpikeMessage, SynapticImpactMessage

using AllMessages = boost::mp11::mp_list<ALL_MESSAGES>;

using MessageVariant = boost::mp11::mp_rename<AllMessages, ::cuda::std::variant>;


__global__ void get_message_kernel(const MessageVariant *var, int *type, const void **msg);

namespace detail
{
template<size_t index>
MessageVariant extract_message_by_index(const void *msg_ptr)
{
    return extract<boost::mp11::mp_at_c<AllMessages, index>>(
            reinterpret_cast<const boost::mp11::mp_at_c<AllMessages, index> *>(msg_ptr));
}
}

template<>
MessageVariant extract<MessageVariant>(const MessageVariant *message)
{
    int *type_gpu;
    const void **msg_gpu; // This is a gpu pointer to gpu pointer to gpu message.
    call_and_check(cudaMalloc(&type_gpu, sizeof(int)));
    call_and_check(cudaMalloc(&msg_gpu, sizeof(void *)));
    get_message_kernel<<<1, 1>>>(message, type_gpu, msg_gpu);
    int type;

    // This is a gpu pointer to gpu message. &msg_ptr is a cpu pointer to gpu pointer to gpu message.
    const void *msg_ptr;
    call_and_check(cudaMemcpy(&type, type_gpu, sizeof(int), cudaMemcpyDeviceToHost));
    call_and_check(cudaMemcpy(&msg_ptr, msg_gpu, sizeof(void *), cudaMemcpyDeviceToHost));
    call_and_check(cudaFree(type_gpu));
    call_and_check(cudaFree(msg_gpu));
    // Here we have a type index and a gpu pointer to message.
    MessageVariant result;
    switch(type)
    {
        case 0: result = detail::extract_message_by_index<0>(msg_ptr);
        case 1: result = detail::extract_message_by_index<1>(msg_ptr);
    }
    return result;
}

}  // namespace knp::backends::gpu::cuda

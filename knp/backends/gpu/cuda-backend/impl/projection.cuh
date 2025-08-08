/**
 * @file projection.cuh
 * @brief GPU projection implementation.
 * @kaspersky_support Artiom N.
 * @date 24.02.2025
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

#include <unordered_map>

#include <knp/core/projection.h>
#include <knp/synapse-traits/all_traits.h>

#include "uid.cuh"


/**
 * @brief Namespace for single-threaded backend.
 */
namespace knp::backends::gpu::cuda
{

/**
 * @brief The CUDAProjection class is a definition of a CUDA synapses.
 */
template <typename SynapseType>
struct CUDAProjection
{
    /**
     * @brief Type of the projection synapses.
     */
    using ProjectionSynapseType = SynapseType;
    /**
     * @brief Projection of synapses with the specified synapse type.
     */
    using ProjectionType = CUDAProjection<SynapseType>;
    /**
     * @brief Parameters of the specified synapse type.
     */
    using SynapseParameters = typename synapse_traits::synapse_parameters<SynapseType>;

    /**
     * @brief Synapse description structure that contains synapse parameters and indexes of the associated neurons.
     */
    using Synapse = thrust::tuple<SynapseParameters, uint32_t, uint32_t>;

    /**
     * @brief Constructor.
     * @param projection source projection.
     */
    explicit CUDAProjection(const knp::core::Projection<SynapseType> &projection)
        : uid_(uid_to_cuda(projection.get_uid())),
          presynaptic_uid_(uid_to_cuda(projection.get_presynaptic())),
          postsynaptic_uid_(uid_to_cuda(projection.get_postsynaptic())),
          is_locked_(thrust::device_malloc<bool>(1))
    {
        *is_locked_ = projection.is_locked();
    }
    /**
     * @brief Destructor.
     */
    ~CUDAProjection() { thrust::device_free(is_locked_); }

    /**
     * @brief UID.
     */
    cuda::UID uid_;

    /**
     * @brief UID of the population that sends spikes to the projection (presynaptic population)
     */
    cuda::UID presynaptic_uid_;

    /**
     * @brief UID of the population that receives synapse responses from this projection (postsynaptic population).
     */
    cuda::UID postsynaptic_uid_;

    /**
     * @brief Return `false` if the weight change for synapses is not locked.
     */
    thrust::device_ptr<bool> is_locked_;

    /**
     * @brief Container of synapse parameters.
     */
//    thrust::device_vector<Synapse> synapses_;

    /**
     * @brief Messages container.
     */
    // cppcheck-suppress unusedStructMember
    std::unordered_map<uint64_t, knp::core::messaging::SynapticImpactMessage> messages_;
};

} // namespace knp::backends::gpu::cuda

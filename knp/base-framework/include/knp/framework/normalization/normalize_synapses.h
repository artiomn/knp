/**
 * @file normalize_synapses.h
 * @brief Synapse normalization functions.
 * @kaspersky_support Artiom N.
 * @date 23.07.2025
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

#include <knp/core/core.h>
#include <knp/core/projection.h>
#include <knp/framework/network.h>

#include <functional>


/**
 * @brief Framework namespace.
 */
namespace knp::framework
{

/**
 * @brief Iterate over synapses in projection normalize parameters.
 * @param projection Projection to normalize weights.
 * @param accessor Function used for access synapse field and normalization.
 */
template <typename SynapseType>
void normalize_synapses(
    knp::core::Projection<SynapseType> &projection,
    std::function<void(typename knp::core::Projection<SynapseType>::SynapseParameters &)> accessor)
{
    for (auto &synapse : projection)
    {
        accessor(std::get<knp::core::synapse_data>(synapse));
    }
}


/**
 * @brief Iterate over all synapses in the network and normalize parameters.
 * @param network Network to normalize weights.
 * @param accessor Function used for access synapse field and normalization..
 */
template <typename SynapseType>
void normalize_synapses(
    knp::framework::Network &network,
    std::function<void(typename knp::core::Projection<SynapseType>::SynapseParameters &)> accessor)
{
    for (auto &projection_variant : network.get_projections())
    {
        // cppcheck-suppress constParameterReference
        auto &projection = std::visit(projection_variant, [](auto &proj) { return proj; });
        normalize_synapses(projection, accessor);
    }
}


/**
 * @brief Iterate over all synapses in the network and normalize parameters.
 * @param network Network to normalize synapse fields.
 * @param accessor Function used for access synapse field and normalization.
 * @return copy of the network with normalized synapses.
 */
template <typename SynapseType>
knp::framework::Network normalize_synapses(
    const knp::framework::Network &network,
    std::function<void(typename knp::core::Projection<SynapseType>::SynapseParameters &)> accessor)
{
    auto new_network = network;

    normalize_synapses(new_network, accessor);

    return network;
}

}  // namespace knp::framework

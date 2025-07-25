/**
 * @file normalize_neurons.h
 * @brief Neurons normalization functions.
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
#include <knp/core/population.h>
#include <knp/framework/network.h>

#include <functional>


/**
 * @brief Framework namespace.
 */
namespace knp::framework
{

/**
 * @brief Iterate over neurons in projection normalize parameters.
 * @param population Population to normalize weights.
 * @param accessor Function used for access synapse field and normalization.
 */
template <typename NeuronType>
void normalize_neurons(
    knp::core::Population<NeuronType> &population,
    std::function<void(typename knp::core::Population<NeuronType>::NeuronParameters &)> accessor)
{
    for (auto &neuron : population)
    {
        accessor(neuron);
    }
}


/**
 * @brief Iterate over all neurons in the network and normalize parameters.
 * @param network Network to normalize.
 * @param accessor Function used for access synapse field and normalization..
 */
template <typename NeuronType>
void normalize_neurons(
    knp::framework::Network &network,
    std::function<void(typename knp::core::Population<NeuronType>::NeuronParameters &)> accessor)
{
    for (auto &population_variant : network.get_populations())
    {
        // cppcheck-suppress constParameterReference
        auto &population = std::visit(population_variant, [](auto &pop) { return pop; });
        normalize_neurons(population, accessor);
    }
}


/**
 * @brief Iterate over all neurons in the network and normalize parameters.
 * @param network Network to normalize.
 * @param accessor Function used for access neuron field and normalization.
 * @return copy of the network with normalized neurons.
 */
template <typename NeuronType>
knp::framework::Network normalize_neurons(
    const knp::framework::Network &network,
    std::function<void(typename knp::core::Population<NeuronType>::NeuronParameters &)> accessor)
{
    auto new_network = network;

    normalize_neurons(new_network, accessor);

    return network;
}

}  // namespace knp::framework

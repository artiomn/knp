/**
 * @file neuron_normalizer.h
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

#include "normalizers.h"


/**
 * @brief Framework normalization namespace.
 */
namespace knp::framework::normalization
{

/**
 * @brief Iterate over neurons in population and normalize parameters.
 * @tparam NeuronType Type of the neuron to change parameter.
 * @tparam DataGetter Type of the class containing method to get parameter value.
 * @tparam DataSetter Type of the class containing method to set parameter value.
 * @tparam ValueCorrector Function to change value.
 * @param population Population to normalize neurons fields.
 * @param getter Neuron field getter.
 * @param setter Neuron field setter.
 * @param corrector Function used for the value normalization.
 */
template <
    typename NeuronType, template <typename> typename DataGetter, template <typename> typename DataSetter,
    typename ValueCorrector = knp::framework::normalization::ValueCorrector<double>>
void normalize_neurons(
    knp::core::Population<NeuronType> &population, DataGetter<NeuronType> getter, DataSetter<NeuronType> setter,
    ValueCorrector corrector)
{
    for (auto &neuron : population)
    {
        // TODO: replace with DataGetter<ElementType>, when C++23 will be used.
        setter(neuron, corrector(getter(neuron)));
    }
}


/**
 * @brief Iterate over neurons in the network and normalize parameters.
 * @tparam DataGetter Type of the class containing method to get parameter value.
 * @tparam DataSetter Type of the class containing method to set parameter value.
 * @tparam ValueCorrector Function to change value.
 * @param network Network to normalize neurons parameters.
 * @param corrector Function used for the value normalization.
 */
template <
    template <typename> typename DataGetter, template <typename> typename DataSetter,
    typename ValueCorrector = knp::framework::normalization::ValueCorrector<double>>
void normalize_neurons(knp::framework::Network &network, ValueCorrector corrector)
{
    for (auto &population_variant : network.get_populations())
    {
        std::visit(
            [&corrector](auto &population)
            {
                using NeuronType = typename std::decay_t<decltype(population)>::PopulationNeuronType;
                normalize_neurons<NeuronType, DataGetter, DataSetter>(
                    const_cast<knp::core::Population<NeuronType> &>(population), DataGetter<NeuronType>(),
                    DataSetter<NeuronType>(), corrector);
            },
            population_variant);
    }
}


/**
 * @brief Iterate over all neurons in the network and normalize parameters.
 * @tparam DataGetter Type of the class containing method to get parameter value.
 * @tparam DataSetter Type of the class containing method to set parameter value.
 * @tparam ValueCorrector Function to change value.
 * @param network Network to normalize.
 * @param corrector Function used for the value normalization.
 * @return copy of the network with normalized neurons.
 */
template <
    template <typename> typename DataGetter, template <typename> typename DataSetter,
    typename ValueCorrector = knp::framework::normalization::ValueCorrector<double>>
knp::framework::Network normalize_neurons(const knp::framework::Network &network, const ValueCorrector &corrector)
{
    auto new_network = network;

    normalize_neurons<DataGetter, DataSetter>(new_network, corrector);

    return network;
}

}  // namespace knp::framework::normalization

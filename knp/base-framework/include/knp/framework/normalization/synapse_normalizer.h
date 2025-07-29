/**
 * @file synapse_normalizer.h
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
#include <type_traits>
#include <variant>

#include "normalizers.h"


/**
 * @brief Framework normalization namespace.
 */
namespace knp::framework::normalization
{

/**
 * @brief Iterate over synapses in projection normalize parameters.
 * @tparam SynapseType Type of the synapse to change parameter.
 * @tparam DataGetter Type of the class containing method to get parameter value.
 * @tparam DataSetter Type of the class containing method to set parameter value.
 * @tparam ValueCorrector Function to change value.
 * @param projection Projection to change parameters.
 * @param getter Synapse field getter.
 * @param setter Synapse field setter.
 * @param corrector Function used for the value normalization.
 */
template <
    typename SynapseType, template <typename> typename DataGetter, template <typename> typename DataSetter,
    typename ValueCorrector = knp::framework::normalization::ValueCorrector<double>>
void normalize_synapses(
    knp::core::Projection<SynapseType> &projection, DataGetter<SynapseType> getter, DataSetter<SynapseType> setter,
    ValueCorrector corrector)
{
    for (auto &synapse : projection)
    {
        auto &synapse_data = std::get<knp::core::synapse_data>(synapse);
        // TODO: replace with DataGetter<ElementType>, when C++23 will be used.
        setter(synapse_data, corrector(getter(synapse_data)));
    }
}


/**
 * @brief Iterate over all synapses in the network and normalize parameters.
 * @tparam DataGetter Type of the class containing method to get parameter value.
 * @tparam DataSetter Type of the class containing method to set parameter value.
 * @tparam ValueCorrector Function to change value.
 * @param network Network to normalize weights.
 * @param corrector Function used for the value normalization.
 */
template <
    template <typename> typename DataGetter, template <typename> typename DataSetter,
    typename ValueCorrector = knp::framework::normalization::ValueCorrector<double>>
void normalize_synapses(knp::framework::Network &network, ValueCorrector corrector)
{
    for (auto &projection_variant : network.get_projections())
    {
        std::visit(
            [&corrector](auto &projection)
            {
                using SynapseType = typename std::decay_t<decltype(projection)>::ProjectionSynapseType;
                normalize_synapses<SynapseType, DataGetter, DataSetter>(
                    const_cast<knp::core::Projection<SynapseType> &>(projection), DataGetter<SynapseType>(),
                    DataSetter<SynapseType>(), corrector);
            },
            projection_variant);
    }
}


/**
 * @brief Iterate over all synapses in the network and normalize parameters.
 * @tparam DataGetter Type of the class containing method to get parameter value.
 * @tparam DataSetter Type of the class containing method to set parameter value.
 * @tparam ValueCorrector Function to change value.
 * @param network Network to normalize synapse fields.
 * @param corrector Function used for the value normalization.
 * @return copy of the network with normalized synapses.
 */
template <
    template <typename> typename DataGetter, template <typename> typename DataSetter,
    typename ValueCorrector = knp::framework::normalization::ValueCorrector<double>>
knp::framework::Network normalize_synapses(const knp::framework::Network &network, const ValueCorrector &corrector)
{
    auto new_network = network;

    normalize_synapses<DataGetter, DataSetter>(new_network, corrector);

    return network;
}

}  // namespace knp::framework::normalization

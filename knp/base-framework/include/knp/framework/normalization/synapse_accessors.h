/**
 * @file synapse_accessors.h
 * @brief Synapse field accessor functions.
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

#include <knp/core/projection.h>
#include <knp/framework/normalization/normalizers.h>
#include <knp/synapse-traits/all_traits.h>


/**
 * @brief Framework normalization namespace.
 */
namespace knp::framework::normalization
{

/**
 * @brief The WeightAccessor class need to access `weight_` synapse field.
 * @tparam SynapseType Type of the synapse to correct.
 */
template <typename SynapseType>
struct WeightAccessor
{
    /**
     * @brief Synapse parameters type.
     */
    using SynapseParametersType = typename knp::core::Projection<SynapseType>::SynapseParameters;
    /**
     * @brief Synapse weight type.
     */
    using WeightType = decltype(SynapseParametersType::weight_);

    /**
     * @brief return weight.
     * @param syn_params synapse parameters.
     * @return synapse weight value.
     */
    WeightType operator()(const SynapseParametersType &syn_params) const { return syn_params.weight_; }

    /**
     * @brief perfrorms weight correction.
     * @param syn_params synapse parameters.
     * @param weight synapse weight.
     */
    void operator()(SynapseParametersType &syn_params, const WeightType &weight) const { syn_params.weight_ = weight; }
};

}  // namespace knp::framework::normalization

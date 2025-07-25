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
 * @brief Framework namespace.
 */
namespace knp::framework
{

/**
 * @brief The WeightAccessor class
 */
class WeightAccessor
{
public:
    /**
     * @brief Synapse weight value type.
     */
    using ValueType = float;
    /**
     * @brief Value corrector type.
     */
    using VCType = knp::framework::ValueCorrector<ValueType>;

public:
    /**
     * @brief WeightAccessor constructor.
     * @param value_corrector normalizer function.
     */
    explicit WeightAccessor(VCType value_corrector) : value_corrector_(value_corrector) {}

    /**
     * @brief operator () perfrorms weight correction.
     * @param synapse real synapse data.
     */
    template <typename SynapseType>
    void operator()(typename knp::core::Projection<typename SynapseType>::SynapseParameters& synapse)
    {
        std::cout << synapse.weight_ << std::endl;
        synapse.weight_ = value_corrector_(synapse.weight_);
    }

private:
    VCType value_corrector_;
};

}  // namespace knp::framework

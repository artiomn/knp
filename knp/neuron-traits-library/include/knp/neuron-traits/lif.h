/**
 * @file lif.h
 * @brief LIF neuron type traits.
 * @kaspersky_support David P.
 * @date 29.04.2026
 * @license Apache 2.0
 * @copyright © 2026 AO Kaspersky Lab
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

#include <cstdint>

#include "type_traits.h"


/**
 * @brief Namespace for neuron traits.
 */
namespace knp::neuron_traits
{

/**
 * @brief LIF neuron.
 * @note Use as a template parameter only.
 */
struct LIFNeuron;


/**
 * @brief Structure for LIF neuron default values.
 */
template <>
struct default_values<LIFNeuron>
{
    constexpr static float potential_ = 0;

    constexpr static float activation_threshold_ = 1;

    constexpr static float leak_coefficient_ = 1;

    constexpr static uint32_t refract_counter_ = 0;

    constexpr static uint32_t refract_period_ = 0;
};


/**
 * @brief Structure for LIF neuron parameters.
 */
template <>
struct neuron_parameters<LIFNeuron>
{
    float potential_ = default_values<LIFNeuron>::potential_;

    float activation_threshold_ = default_values<LIFNeuron>::activation_threshold_;

    float leak_coefficient_ = default_values<LIFNeuron>::leak_coefficient_;

    float refract_counter_ = default_values<LIFNeuron>::refract_counter_;

    float refract_period_ = default_values<LIFNeuron>::refract_period_;
};

}  // namespace knp::neuron_traits

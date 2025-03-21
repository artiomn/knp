/**
 * @file stdp_type_traits.h
 * @brief STDP associations and type traits.
 * @kaspersky_support Artiom N.
 * @date 19.09.2023
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

#include <set>

#include "delta.h"
#include "stdp_add_rule.h"
#include "stdp_synaptic_resource_rule.h"

/**
 * @brief Namespace for synapse traits.
 */
namespace knp::synapse_traits
{

/**
 * @brief Delta synapse with STDP additive rule type.
 */
using AdditiveSTDPDeltaSynapse = STDP<STDPAdditiveRule, DeltaSynapse>;


/**
 * @brief Delta synapse with STDP synaptic resource rule type.
 */
using SynapticResourceSTDPDeltaSynapse = STDP<STDPSynapticResourceRule, DeltaSynapse>;


}  // namespace knp::synapse_traits

/**
 * @file altai_lif_population.h
 * @kaspersky_support Vartenkov A.
 * @date 01.04.2025
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
#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/messaging.h>
#include <knp/core/population.h>

#include <optional>

#include "impl/altai_lif_population_impl.h"
#include "impl/lif_population_impl.h"


/**
 * @brief Namespace for CPU backends.
 */
namespace knp::backends::cpu
{
template <class LifNeuron>
std::optional<knp::core::messaging::SpikeMessage> calculate_lif_population(
    knp::core::Population<LifNeuron> &population, knp::core::MessageEndpoint &endpoint, size_t step_n)
{
    return calculate_lif_population_impl(population, endpoint, step_n);
}
}  // namespace knp::backends::cpu

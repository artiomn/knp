/**
 * @file template_specs.h
 * @brief Header for multi-threaded CPU backend-specific template specializations.
 * @kaspersky_support Vartenkov A.
 * @date 16.06.2025
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

#include <knp/backends/cpu-library/impl/blifat_population_impl.h>
#include <knp/backends/cpu-multi-threaded/backend.h>

/**
 * @brief Namespace for CPU backends.
 */
namespace knp::backends::cpu
{
template <>
void finalize_population<
    knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron,
    multi_threaded_cpu::MultiThreadedCPUBackend::ProjectionContainer>(
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population,
    const knp::core::messaging::SpikeMessage &message,
    multi_threaded_cpu::MultiThreadedCPUBackend::ProjectionContainer &projections, knp::core::Step step);
}  // namespace knp::backends::cpu

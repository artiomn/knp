/**
 * @file template_specs.cpp
 * @brief Multi-threaded CPU backend-specific template specializations.
 * @kaspersky_support Vartenkov A.
 * @date 11.06.2025
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
#include <knp/backends/cpu-library/impl/blifat_population_impl.h>
#include <knp/backends/cpu-library/impl/synaptic_resource_stdp_impl.h>
#include <knp/backends/cpu-multi-threaded/backend.h>


namespace knp::backends::cpu
{
template <>
void finalize_population<
    knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron,
    multi_threaded_cpu::MultiThreadedCPUBackend::ProjectionContainer>(
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population,
    const knp::core::messaging::SpikeMessage &message,
    multi_threaded_cpu::MultiThreadedCPUBackend::ProjectionContainer &projections, knp::core::Step step)
{
    using SynapseType = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
    std::vector<knp::core::Projection<SynapseType> *> working_projections;
    constexpr uint64_t type_index =
        boost::mp11::mp_find<backends::multi_threaded_cpu::MultiThreadedCPUBackend::SupportedSynapses, SynapseType>();

    for (auto &projection : projections)
    {
        if (projection.arg_.index() != type_index)
        {
            continue;
        }

        auto *projection_ptr = &(std::get<type_index>(projection.arg_));
        if (projection_ptr->is_locked())
        {
            continue;
        }

        if (projection_ptr->get_postsynaptic() == population.get_uid())
        {
            working_projections.push_back(projection_ptr);
        }
    }

    do_STDP_resource_plasticity<knp::neuron_traits::BLIFATNeuron, knp::synapse_traits::DeltaSynapse>(
        population, working_projections, message, step);
}
}  //namespace knp::backends::cpu

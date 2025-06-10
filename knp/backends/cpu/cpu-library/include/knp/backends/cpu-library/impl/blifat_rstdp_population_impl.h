//
// Created by vartenkov on 29.04.25.
//

#pragma once

#include <knp/neuron-traits/stdp_synaptic_resource_rule.h>

#include "blifat_population_impl.h"
#include "synaptic_resource_stdp_impl.h"


namespace knp::backends::cpu
{
/**
 * Do synaptic resource plasticity.
 */
//template <>
//void finalize_population<knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron>,
//    knp::backends::multi_threaded_cpu::MultiThreadedCPUBackend::ProjectionContainer>(
//        knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population,
//        const knp::core::messaging::SpikeMessage &message,
//        knp::backends::multi_threaded_cpu::MultiThreadedCPUBackend::ProjectionContainer &projections,
//        knp::core::Step step)
//{
//    using NeuronType = knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron;
//    using ProjContainer = knp::backends::multi_threaded_cpu::MultiThreadedCPUBackend::ProjectionContainer;
//    auto working_projections = find_projection_by_type_and_postsynaptic<
//            NeuronType, ProjContainer>(
//            projections, population.get_uid(), true);
//    do_STDP_resource_plasticity(population, working_projections, message, step);
// }
}  // namespace knp::backends::cpu

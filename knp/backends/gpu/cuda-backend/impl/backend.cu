/**
 * @file backend.cu
 * @brief CUDABackend backend class implementation.
 * @kaspersky_support Artiom N.
 * @date 24.02.2025
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


#include <knp/backends/gpu-cuda/backend.h>
#include <knp/devices/gpu_cuda.h>
#include <knp/meta/assert_helpers.h>
#include <knp/meta/stringify.h>
#include <knp/meta/variant_helpers.h>

#include <spdlog/spdlog.h>

#include <vector>

#include <boost/mp11.hpp>


namespace knp::backends::gpu
{

CUDABackend::CUDABackend()
{
    SPDLOG_INFO("Single-threaded CPU backend instance created.");
}


std::shared_ptr<CUDABackend> CUDABackend::create()
{
    SPDLOG_DEBUG("Creating single-threaded CPU backend instance...");
    return std::make_shared<CUDABackend>();
}


std::vector<std::string> CUDABackend::get_supported_neurons() const
{
    return knp::meta::get_supported_type_names<knp::neuron_traits::AllNeurons, SupportedNeurons>(
        knp::neuron_traits::neurons_names);
}


std::vector<std::string> CUDABackend::get_supported_synapses() const
{
    return knp::meta::get_supported_type_names<knp::synapse_traits::AllSynapses, SupportedSynapses>(
        knp::synapse_traits::synapses_names);
}


std::vector<size_t> CUDABackend::get_supported_projection_indexes() const
{
    return knp::meta::get_supported_type_indexes<core::AllProjections, SupportedProjections>();
}


std::vector<size_t> CUDABackend::get_supported_population_indexes() const
{
    return knp::meta::get_supported_type_indexes<core::AllPopulations, SupportedPopulations>();
}


template <typename AllVariants, typename SupportedVariants>
SupportedVariants convert_variant(const AllVariants &input)
{
    SupportedVariants result = std::visit([](auto &&arg) { return arg; }, input);
    return result;
}


void CUDABackend::_step()
{
    SPDLOG_DEBUG("Starting step #{}...", get_step());
    get_message_bus().route_messages();
    get_message_endpoint().receive_all_messages();
    // Calculate populations. This is the same as inference.
    for (auto &population : populations_)
    {
        std::visit(
            [this](auto &arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (
                    boost::mp11::mp_find<SupportedPopulations, T>{} == boost::mp11::mp_size<SupportedPopulations>{})
                {
                    static_assert(
                        knp::meta::always_false_v<T>,
                        "Population is not supported by the single-threaded CPU backend.");
                }
                auto message_opt = calculate_population(arg);
            },
            population);
    }

    // Continue inference.
    get_message_bus().route_messages();
    get_message_endpoint().receive_all_messages();
    // Calculate projections.
    for (auto &projection : projections_)
    {
        std::visit(
            [this, &projection](auto &arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (
                    boost::mp11::mp_find<SupportedProjections, T>{} == boost::mp11::mp_size<SupportedProjections>{})
                {
                    static_assert(
                        knp::meta::always_false_v<T>,
                        "Projection is not supported by the single-threaded CPU backend.");
                }
                calculate_projection(arg, projection.messages_);
            },
            projection.arg_);
    }

    get_message_bus().route_messages();
    get_message_endpoint().receive_all_messages();
    auto step = gad_step();
    // Need to suppress "Unused variable" warning.
    (void)step;
    SPDLOG_DEBUG("Step finished #{}.", step);
}


void CUDABackend::load_populations(const std::vector<PopulationVariants> &populations)
{
    SPDLOG_DEBUG("Loading populations [{}]...", populations.size());
    populations_.clear();
    populations_.reserve(populations.size());

    for (const auto &population : populations)
    {
        populations_.push_back(population);
    }
    SPDLOG_DEBUG("All populations loaded.");
}


void CUDABackend::load_projections(const std::vector<ProjectionVariants> &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());
    projections_.clear();
    projections_.reserve(projections.size());

    for (const auto &projection : projections)
    {
        projections_.push_back(ProjectionWrapper{projection});
    }

    SPDLOG_DEBUG("All projections loaded.");
}


void CUDABackend::load_all_projections(const std::vector<knp::core::AllProjectionsVariant> &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());
    knp::meta::load_from_container<SupportedProjections>(projections, projections_);
    SPDLOG_DEBUG("All projections loaded.");
}


void CUDABackend::load_all_populations(const std::vector<knp::core::AllPopulationsVariant> &populations)
{
    SPDLOG_DEBUG("Loading populations [{}]...", populations.size());
    knp::meta::load_from_container<SupportedPopulations>(populations, populations_);
    SPDLOG_DEBUG("All populations loaded.");
}


std::vector<std::unique_ptr<knp::core::Device>> CUDABackend::get_devices() const
{
    std::vector<std::unique_ptr<knp::core::Device>> result;
    auto &&processors{knp::devices::gpu::cuda::list_processors()};

    result.reserve(processors.size());

    for (auto &&gpu : processors)
    {
        SPDLOG_DEBUG("GPU \"{}\".", gpu.get_name());
        result.push_back(std::make_unique<knp::devices::gpu::CUDA>(std::move(gpu)));
    }

    SPDLOG_DEBUG("NVIDIA GPU count = {}.", result.size());
    return result;
}


void CUDABackend::_init()
{
    SPDLOG_DEBUG("Initializing CUDABackend backend...");

    // knp::backends::cpu::init(projections_, get_message_endpoint());

    SPDLOG_DEBUG("Initialization finished.");
}


std::optional<core::messaging::SpikeMessage> CUDABackend::calculate_population(
    core::Population<knp::neuron_traits::BLIFATNeuron> &population)
{
    SPDLOG_TRACE("Calculate BLIFAT population {}.", std::string(population.get_uid()));
    return std::nullopt;  // knp::backends::cpu::calculate_blifat_population(population, get_message_endpoint(),
                          // get_step());
}


std::optional<core::messaging::SpikeMessage> CUDABackend::calculate_population(
    knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population)
{
    SPDLOG_TRACE("Calculate resource-based STDP-compatible BLIFAT population {}.", std::string(population.get_uid()));
    return std::nullopt;
}


void CUDABackend::calculate_projection(
    knp::core::Projection<knp::synapse_traits::DeltaSynapse> &projection, SynapticMessageQueue &message_queue)
{
    SPDLOG_TRACE("Calculate delta synapse projection {}.", std::string(projection.get_uid()));
    // knp::backends::cpu::calculate_delta_synapse_projection(
    //    projection, get_message_endpoint(), message_queue, get_step());
}


void CUDABackend::calculate_projection(
    knp::core::Projection<knp::synapse_traits::AdditiveSTDPDeltaSynapse> &projection,
    SynapticMessageQueue &message_queue)
{
    SPDLOG_TRACE("Calculate AdditiveSTDPDelta synapse projection {}.", std::string(projection.get_uid()));
}


void CUDABackend::calculate_projection(
    knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> &projection,
    SynapticMessageQueue &message_queue)
{
    SPDLOG_TRACE("Calculate STDPSynapticResource synapse projection {}.", std::string(projection.get_uid()));
}


CUDABackend::PopulationIterator CUDABackend::begin_populations()
{
    return PopulationIterator{populations_.begin()};
}


CUDABackend::PopulationConstIterator CUDABackend::begin_populations() const
{
    return {populations_.cbegin()};
}


CUDABackend::PopulationIterator CUDABackend::end_populations()
{
    return PopulationIterator{populations_.end()};
}


CUDABackend::PopulationConstIterator CUDABackend::end_populations() const
{
    return populations_.cend();
}


CUDABackend::ProjectionIterator CUDABackend::begin_projections()
{
    return ProjectionIterator{projections_.begin()};
}


CUDABackend::ProjectionConstIterator CUDABackend::begin_projections() const
{
    return projections_.cbegin();
}


CUDABackend::ProjectionIterator CUDABackend::end_projections()
{
    return ProjectionIterator{projections_.end()};
}


CUDABackend::ProjectionConstIterator CUDABackend::end_projections() const
{
    return projections_.cend();
}


BOOST_DLL_ALIAS(knp::backends::gpu::CUDABackend::create, create_knp_backend)

}  // namespace knp::backends::gpu

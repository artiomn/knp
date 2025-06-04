/**
 * @file backend.cu
 * @brief CUDABackendImpl backend class implementation.
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


#include "backend_impl.cuh"
#include <knp/devices/gpu_cuda.h>
#include <knp/meta/assert_helpers.h>
#include <knp/meta/stringify.h>
#include <knp/meta/variant_helpers.h>

#include <spdlog/spdlog.h>

#include <limits>
#include <vector>

#include <boost/mp11.hpp>


namespace knp::backends::gpu
{


template <typename AllVariants, typename SupportedVariants>
SupportedVariants convert_variant(const AllVariants &input)
{
    SupportedVariants result = std::visit([](auto &&arg) { return arg; }, input);
    return result;
}


template <class ProjectionType>
constexpr bool is_forcing()
{
    return false;
}


template <>
constexpr bool is_forcing<knp::core::Projection<synapse_traits::DeltaSynapse>>()
{
    return true;
}


void CUDABackendImpl::_step()
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
                        "Population is not supported by the CUDA backend.");
                }
                auto message_opt = calculate_population(arg, get_step());
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
                        "Projection is not supported by the CUDA backend.");
                }
//                calculate_projection(arg, projection.messages_);
            },
            projection);
    }

    get_message_bus().route_messages();
    get_message_endpoint().receive_all_messages();
    auto step = gad_step();
    // Need to suppress "Unused variable" warning.
    (void)step;
    SPDLOG_DEBUG("Step finished #{}.", step);
}


void CUDABackendImpl::load_populations(const std::vector<PopulationVariants> &populations)
{
    SPDLOG_DEBUG("Loading populations [{}]...", populations.size());

    device_populations_.clear();
    device_populations_.reserve(populations.size());

    for (const auto &population : populations)
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
                        "Population is not supported by the CUDA backend.");
                }

                device_populations_.push_back(CUDAPopulation<typename T::PopulationNeuronType>(arg));
            },
            population);
    }
    SPDLOG_DEBUG("All populations loaded.");
}


void CUDABackendImpl::load_projections(const std::vector<ProjectionVariants> &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());
    projections_.clear();
    projections_.reserve(projections.size());

    for (const auto &projection : projections)
    {
        projections_.push_back(projection);
    }

    device_projections_.clear();
    device_projections_.reserve(projections.size());

    for (const auto &projection : projections)
    {
        std::visit(
            [this](auto &arg)
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (
                    boost::mp11::mp_find<SupportedProjections, T>{} == boost::mp11::mp_size<SupportedProjections>{})
                {
                    static_assert(
                        knp::meta::always_false_v<T>,
                        "Projection is not supported by the CUDA backend.");
                }

                device_projections_.push_back(CUDAProjection<typename T::ProjectionSynapseType>(arg));
//                    .neurons_ = arg.get_neurons_parameters()
            },
            projection);
    }

    SPDLOG_DEBUG("All projections loaded.");
}


void CUDABackendImpl::load_all_projections(const std::vector<knp::core::AllProjectionsVariant> &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());
    knp::meta::load_from_container<SupportedProjections>(projections, projections_);
    SPDLOG_DEBUG("All projections loaded.");
}


void CUDABackendImpl::load_all_populations(const std::vector<knp::core::AllPopulationsVariant> &populations)
{
    SPDLOG_DEBUG("Loading populations [{}]...", populations.size());
    knp::meta::load_from_container<SupportedPopulations>(populations, populations_);
    SPDLOG_DEBUG("All populations loaded.");
}


std::vector<std::unique_ptr<knp::core::Device>> CUDABackendImpl::get_devices() const
{
    std::vector<std::unique_ptr<knp::core::Device>> result;
    auto &&processors{knp::devices::gpu::list_cuda_processors()};
    result.reserve(processors.size());
    for (auto &&gpu : processors)
    {
        SPDLOG_DEBUG("GPU \"{}\".", gpu.get_name());
        result.push_back(std::make_unique<knp::devices::gpu::CUDA>(std::move(gpu)));
    }

    SPDLOG_DEBUG("NVIDIA GPU count = {}.", result.size());
    return result;
}


void CUDABackendImpl::_init()
{
    SPDLOG_DEBUG("Initializing CUDABackendImpl backend...");

    // knp::backends::cpu::init(projections_, get_message_endpoint());

    SPDLOG_DEBUG("Initialization finished.");
}


__device__ std::optional<knp::backends::gpu::cuda::SpikeMessage> CUDABackendImpl::calculate_population(
    core::Population<knp::neuron_traits::BLIFATNeuron> &population,
    uint64_t step_n)
{
    std::vector<core::messaging::SynapticImpactMessage> messages =
        message_bus_.unload_messages<core::messaging::SynapticImpactMessage>(population.get_uid());

    for (size_t i = 0; i < population.size(); ++i)
    {
        auto &neuron = population[i];
        ++neuron.n_time_steps_since_last_firing_;

        neuron.dynamic_threshold_ *= neuron.threshold_decay_;
        neuron.postsynaptic_trace_ *= neuron.postsynaptic_trace_decay_;
        neuron.inhibitory_conductance_ *= neuron.inhibitory_conductance_decay_;

        /*
        if constexpr (has_dopamine_plasticity<BlifatLikeNeuron>())
        {
            neuron.dopamine_value_ = 0.0;
            neuron.is_being_forced_ = false;
        }
        */

        if (neuron.bursting_phase_ && !--neuron.bursting_phase_)
        {
            neuron.potential_ = neuron.potential_ * neuron.potential_decay_ + neuron.reflexive_weight_;
        }
        else
        {
            neuron.potential_ *= neuron.potential_decay_;
        }
        neuron.pre_impact_potential_ = neuron.potential_;
    }

    // process_inputs(population, messages);
    for (const auto &message : messages)
    {
        for (const auto &impact : message.impacts_)
        {
            auto &neuron = population[impact.postsynaptic_neuron_index_];

            // impact_neuron<BlifatLikeNeuron>(neuron, impact.synapse_type_, impact.impact_value_);
            switch (impact.synapse_type_)
            {
                case knp::synapse_traits::OutputType::EXCITATORY:
                    neuron.potential_ += impact.impact_value_;
                    break;
                case knp::synapse_traits::OutputType::INHIBITORY_CURRENT:
                    neuron.potential_ -= impact.impact_value_;
                    break;
                case knp::synapse_traits::OutputType::INHIBITORY_CONDUCTANCE:
                    neuron.inhibitory_conductance_ += impact.impact_value_;
                    break;
                case knp::synapse_traits::OutputType::DOPAMINE:
                    neuron.dopamine_value_ += impact.impact_value_;
                    break;
                case knp::synapse_traits::OutputType::BLOCKING:
                    neuron.total_blocking_period_ = static_cast<unsigned int>(impact.impact_value_);
                    break;
            }

            /*
            if constexpr (has_dopamine_plasticity<BlifatLikeNeuron>())
            {
                if (impact.synapse_type_ == synapse_traits::OutputType::EXCITATORY)
                {
                    neuron.is_being_forced_ |= message.is_forcing_;
                }
            }
            */
        }
    }
    //

    knp::core::messaging::SpikeData neuron_indexes;

    // calculate_neurons_post_input_state(population, neuron_indexes);
    for (size_t index = 0; index < population.size(); ++index)
    {
        bool spike = false;
        auto &neuron = population[index];

        if (neuron.total_blocking_period_ <= 0)
        {
            // TODO: Make it more readable, don't be afraid to use if operators.
            // Restore potential that the neuron had before impacts.
            neuron.potential_ = neuron.pre_impact_potential_;
            bool was_negative = neuron.total_blocking_period_ < 0;
            // If it is negative, increase by 1.
            neuron.total_blocking_period_ += was_negative;
            // If it is now zero, but was negative before, increase it to max, else leave it as is.
            neuron.total_blocking_period_ +=
                std::numeric_limits<int64_t>::max() * ((neuron.total_blocking_period_ == 0) && was_negative);
        }
        else
        {
            neuron.total_blocking_period_ -= 1;
        }

        if (neuron.inhibitory_conductance_ < 1.0)
        {
            neuron.potential_ -=
                (neuron.potential_ - neuron.reversal_inhibitory_potential_) * neuron.inhibitory_conductance_;
        }
        else
        {
            neuron.potential_ = neuron.reversal_inhibitory_potential_;
        }

        if ((neuron.n_time_steps_since_last_firing_ > neuron.absolute_refractory_period_) &&
            (neuron.potential_ >= neuron.activation_threshold_ + neuron.dynamic_threshold_))
        {
            // Spike.
            neuron.dynamic_threshold_ += neuron.threshold_increment_;
            neuron.postsynaptic_trace_ += neuron.postsynaptic_trace_increment_;

            neuron.potential_ = neuron.potential_reset_value_;
            neuron.bursting_phase_ = neuron.bursting_period_;
            neuron.n_time_steps_since_last_firing_ = 0;
            spike = true;
        }

        if (neuron.potential_ < neuron.min_potential_)
        {
            neuron.potential_ = neuron.min_potential_;
        }

        if (spike)
        {
            neuron_indexes.push_back(index);
        }
    }

    if (!neuron_indexes.empty())
    {
        knp::backends::gpu::cuda::SpikeMessage res_message{{population.get_uid(), step_n}, neuron_indexes};
        message_bus_.send_message(res_message);
        return res_message;
    }
    return {};
}


// std::optional<core::messaging::SpikeMessage> CUDABackendImpl::calculate_population(
//     knp::core::Population<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population)
//{
//    SPDLOG_TRACE("Calculate resource-based STDP-compatible BLIFAT population {}.", std::string(population.get_uid()));
//    return std::nullopt;
//}


__device__ void CUDABackendImpl::calculate_projection(
    knp::core::Projection<knp::synapse_traits::DeltaSynapse> &projection, SynapticMessageQueue &message_queue)
{
    auto messages = message_bus_.unload_messages<knp::backends::gpu::cuda::SpikeMessage>(projection.get_uid());

    // auto out_iter = calculate_delta_synapse_projection_data(projection, messages, future_messages, get_step());
    //
    // using SynapseType = typename ProjectionType::ProjectionSynapseType;
    // WeightUpdateSTDP<SynapseType>::init_projection(projection, messages, step_n);

    for (const auto &message : messages)
    {
        const auto &message_data = message.neuron_indexes_;
        for (const auto &spiked_neuron_index : message_data)
        {
            auto synapses = projection.find_synapses(spiked_neuron_index,
                                                     typename decltype(projection)::Search::by_presynaptic);
            for (auto synapse_index : synapses)
            {
                auto &synapse = projection[synapse_index];
                // WeightUpdateSTDP<SynapseType>::init_synapse(std::get<core::synapse_data>(synapse), step_n);
                const auto &synapse_params = std::get<core::synapse_data>(synapse);

                // The message is sent on step N - 1, received on step N.
                size_t future_step = synapse_params.delay_ + step_n - 1;
                knp::backends::gpu::cuda::SynapticImpact impact{
                    synapse_index, synapse_params.weight_, synapse_params.output_type_,
                    static_cast<uint32_t>(std::get<core::source_neuron_id>(synapse)),
                    static_cast<uint32_t>(std::get<core::target_neuron_id>(synapse))};

                auto iter = future_messages.find(future_step);
                if (iter != future_messages.end())
                {
                    iter->second.impacts_.push_back(impact);
                }
                else
                {
                    knp::core::messaging::SynapticImpactMessage message_out{
                        {projection.get_uid(), step_n},
                        projection.get_presynaptic(),
                        projection.get_postsynaptic(),
                        is_forcing<ProjectionType>(),
                        {impact}};
                    future_messages.insert(std::make_pair(future_step, message_out));
                }
            }
        }
    }
    // WeightUpdateSTDP<SynapseType>::modify_weights(projection);
    return future_messages.find(step_n);
    //

    if (out_iter != future_messages.end())
    {
        // Send a message and remove it from the queue.
        message_bus_.send_message(out_iter->second);
        future_messages.erase(out_iter);
    }

    // knp::backends::cpu::calculate_delta_synapse_projection(
    //    projection, get_message_endpoint(), message_queue, get_step());
}


//void CUDABackendImpl::calculate_projection(
//    knp::core::Projection<knp::synapse_traits::AdditiveSTDPDeltaSynapse> &projection,
//    SynapticMessageQueue &message_queue)
//{
//    SPDLOG_TRACE("Calculate AdditiveSTDPDelta synapse projection {}.", std::string(projection.get_uid()));
//}


//void CUDABackendImpl::calculate_projection(
//    knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> &projection,
//    SynapticMessageQueue &message_queue)
//{
//    SPDLOG_TRACE("Calculate STDPSynapticResource synapse projection {}.", std::string(projection.get_uid()));
//}


CUDABackendImpl::PopulationIterator CUDABackendImpl::begin_populations()
{
    return PopulationIterator{populations_.begin()};
}


CUDABackendImpl::PopulationConstIterator CUDABackendImpl::begin_populations() const
{
    return {populations_.cbegin()};
}


CUDABackendImpl::PopulationIterator CUDABackendImpl::end_populations()
{
    return PopulationIterator{populations_.end()};
}


CUDABackendImpl::PopulationConstIterator CUDABackendImpl::end_populations() const
{
    return populations_.cend();
}


CUDABackendImpl::ProjectionIterator CUDABackendImpl::begin_projections()
{
    return ProjectionIterator{projections_.begin()};
}


CUDABackendImpl::ProjectionConstIterator CUDABackendImpl::begin_projections() const
{
    return projections_.cbegin();
}


CUDABackendImpl::ProjectionIterator CUDABackendImpl::end_projections()
{
    return ProjectionIterator{projections_.end()};
}


CUDABackendImpl::ProjectionConstIterator CUDABackendImpl::end_projections() const
{
    return projections_.cend();
}


}  // namespace knp::backends::gpu

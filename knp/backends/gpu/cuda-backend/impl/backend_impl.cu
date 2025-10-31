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


#include <knp/devices/gpu_cuda.h>
#include <knp/meta/assert_helpers.h>
#include <knp/meta/stringify.h>
#include <knp/meta/variant_helpers.h>

#include <spdlog/spdlog.h>

#include <limits>
#include <vector>

#include <boost/mp11.hpp>

#include <algorithm>

#include "backend_impl.cuh"
#include "projection.cuh"
#include "population.cuh"

#include "cuda_lib/get_blocks_config.cuh"
#include "cuda_lib/vector.cuh"


namespace knp::backends::gpu::cuda
{

// helper type for the visitor.
template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };
// explicit deduction guide.
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;


template <class ProjectionType>
inline bool is_forcing()
{
    return false;
}


template <>
inline bool is_forcing<cuda::CUDAProjection<synapse_traits::DeltaSynapse>>()
{
    return true;
}


__global__ void calculate_populations_kernel(cuda::CUDABackendImpl *backend,
                                             typename cuda::CUDABackendImpl::PopulationContainer &populations,
                                             std::uint64_t step)
{
    // Calculate populations. This is the same as inference.
    for (auto &population : populations)
    {
//                knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::SynapticImpactMessage> messages;
//                auto arg = ::std::get<CUDAPopulation<knp::neuron_traits::BLIFATNeuron>>(population);
//                auto message_opt = cuda::CUDABackendImpl::calculate_population(arg, messages, step);
/*        ::cuda::std::visit(
            [backend, step](auto &arg)
            {
                using T = std::decay_t<decltype(arg)>;

                knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::SynapticImpactMessage> messages;
                auto message_opt = cuda::CUDABackendImpl::calculate_population(arg, messages, step);
            },
            population);*/
    }
}


__global__ void calculate_projections_kernel(cuda::CUDABackendImpl *backend,
                                             typename cuda::CUDABackendImpl::ProjectionContainer &projections,
                                             std::uint64_t step)
{
    // Calculate projections.
    // using namespace std::placeholders;

    for (auto &projection : projections)
    {
        knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::SpikeMessage> messages;
        // TODO: Uncomment

            auto arg = ::cuda::std::get<CUDAProjection<knp::synapse_traits::DeltaSynapse>>(projection);
            cuda::CUDABackendImpl::calculate_projection(arg, messages, step);

/*
        ::cuda::std::visit([backend, step](auto &arg)
        {
            using T = std::decay_t<decltype(arg)>;

            knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::SpikeMessage> messages;
            cuda::CUDABackendImpl::calculate_projection(arg, messages, step);
        }, projection);
*/
    }
}


void CUDABackendImpl::calculate_populations(std::uint64_t step)
{
    // Calculate populations. This is the same as inference.
    // Calculate projections.
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(device_populations_.size());

    calculate_populations_kernel<<<num_blocks, num_threads>>>(this, device_populations_, step);
    cudaDeviceSynchronize();
}


void CUDABackendImpl::calculate_projections(std::uint64_t step)
{
    // Calculate projections.
    auto [num_blocks, num_threads] = device_lib::get_blocks_config(device_projections_.size());

    calculate_projections_kernel<<<num_blocks, num_threads>>>(this, device_projections_, step);
    cudaDeviceSynchronize();
}


void CUDABackendImpl::load_populations(const knp::backends::gpu::CUDABackend::PopulationContainer &populations)
{
    SPDLOG_DEBUG("Loading populations [{}]...", populations.size());

    device_populations_.clear();
    device_populations_.reserve(populations.size());

    for (const auto &population : populations)
    {
        ::std::visit(
            [this](auto &arg)
            {
                using CPUPopulationType = std::decay_t<decltype(arg)>;

                auto pop = CUDAPopulation<typename CPUPopulationType::PopulationNeuronType>(arg);
                device_populations_.push_back(pop);
            },
            population);
    }
    SPDLOG_DEBUG("All populations loaded.");
}


void CUDABackendImpl::load_projections(const knp::backends::gpu::CUDABackend::ProjectionContainer &projections)
{
    SPDLOG_DEBUG("Loading projections [{}]...", projections.size());

    device_projections_.clear();
    device_projections_.reserve(projections.size());

    for (const auto &projection : projections)
    {
        ::std::visit(
            [this](auto &arg)
            {
                using CPUProjectionType = std::decay_t<decltype(arg)>;
                auto proj = CUDAProjection<typename CPUProjectionType::ProjectionSynapseType>(arg);

                device_projections_.push_back(proj);
            },
            projection);
    }

    SPDLOG_DEBUG("All projections loaded.");
}


void CUDABackendImpl::_init()
{
//    SPDLOG_DEBUG("Initializing CUDABackendImpl backend...");

    // knp::backends::cpu::init(projections_, get_message_endpoint());
//    SPDLOG_DEBUG("Initialization finished.");
}


__device__ ::cuda::std::optional<knp::backends::gpu::cuda::SpikeMessage> CUDABackendImpl::calculate_population(
    CUDAPopulation<knp::neuron_traits::BLIFATNeuron> &population,
    knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::SynapticImpactMessage> &messages,
    std::uint64_t step_n)
{
    // TODO rework
    for (size_t i = 0; i < population.neurons_.size(); ++i)
    {
        neuron_traits::neuron_parameters<neuron_traits::BLIFATNeuron> neuron = population.neurons_[i];
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

        population.neurons_[i] = neuron;
    }

    // process_inputs(population, messages);
    for (const cuda::SynapticImpactMessage &message : messages)
    {
        for (size_t i = 0; i < message.impacts_.size(); ++i)
        {
            const auto &impact = message.impacts_[i];

            neuron_traits::neuron_parameters<neuron_traits::BLIFATNeuron> neuron =
                population.neurons_[impact.postsynaptic_neuron_index_];

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

            /*if constexpr (has_dopamine_plasticity<BlifatLikeNeuron>())
            {
                if (impact.synapse_type_ == synapse_traits::OutputType::EXCITATORY)
                {
                    neuron.is_being_forced_ |= message.is_forcing_;
                }
            }*/
            population.neurons_[impact.postsynaptic_neuron_index_] = neuron;
        }
    }

    device_lib::CUDAVector<uint32_t> neuron_indexes;

    // calculate_neurons_post_input_state(population, neuron_indexes);
    for (size_t index = 0; index < population.neurons_.size(); ++index)
    {
        bool spike = false;
        neuron_traits::neuron_parameters<neuron_traits::BLIFATNeuron> neuron = population.neurons_[index];

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

        population.neurons_[index] = neuron;
    }

    if (!neuron_indexes.empty())
    {
        cuda::SpikeMessage res_message
        {
            .header_ = { .sender_uid_ = population.uid_, step_n},
            .neuron_indexes_ = neuron_indexes
        };

//        device_message_bus_.send_message(res_message);
        return res_message;
    }

    return {};
}


/*__device__ ::cuda::std::optional<core::messaging::SpikeMessage> CUDABackendImpl::calculate_population(
    CUDAPopulation<knp::neuron_traits::SynapticResourceSTDPBLIFATNeuron> &population,
    knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::SynapticImpactMessage> &messages,
    std::uint64_t step_n)
{
//    SPDLOG_TRACE("Calculate resource-based STDP-compatible BLIFAT population {}.", std::string(population.get_uid()));
    return std::nullopt;
}*/


__device__ void CUDABackendImpl::calculate_projection(
    CUDAProjection<knp::synapse_traits::DeltaSynapse> &projection,
    knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::SpikeMessage> &messages,
    std::uint64_t step_n)
{
    // Run:
    // knp::backends::cpu::calculate_delta_synapse_projection(
    //    projection, get_message_endpoint(), message_queue, get_step());


    // message_bus_.unload_messages<cuda::SpikeMessage>(projection.uid_, messages);

    // auto out_iter = calculate_delta_synapse_projection_data(projection, messages, future_messages, get_step());
    //
    // using SynapseType = typename ProjectionType::ProjectionSynapseType;
    // WeightUpdateSTDP<SynapseType>::init_projection(projection, messages, step_n);

    for (const knp::backends::gpu::cuda::SpikeMessage message : messages)
    {
        const auto &message_data = message.neuron_indexes_;
        for (size_t i = 0; i < message_data.size(); ++i)
        {
            const auto &spiked_neuron_index = message_data[i];

            for (size_t synapse_index = 0; synapse_index < projection.synapses_.size(); ++synapse_index)
            {
                CUDAProjection<knp::synapse_traits::DeltaSynapse>::Synapse synapse =
                    projection.synapses_[synapse_index];
                if (thrust::get<core::source_neuron_id>(synapse) != spiked_neuron_index) continue;

                // WeightUpdateSTDP<SynapseType>::init_synapse(std::get<core::synapse_data>(synapse), step_n);
                const auto &synapse_params = thrust::get<core::synapse_data>(synapse);

                // The message is sent on step N - 1, received on step N.
                size_t future_step = synapse_params.delay_ + step_n - 1;
                knp::backends::gpu::cuda::SynapticImpact impact{
                    synapse_index, synapse_params.weight_, synapse_params.output_type_,
                    static_cast<uint32_t>(thrust::get<core::source_neuron_id>(synapse)),
                    static_cast<uint32_t>(thrust::get<core::target_neuron_id>(synapse))};

                // ::cuda::std::find_if() is not implemented yet.
                auto iter = projection.messages_.begin();
                for (; iter != projection.messages_.end(); ++iter)
                {
                    if (iter->first == future_step) break;
                }

                if (iter != projection.messages_.end())
                {
                    iter->second.impacts_.push_back(impact);
                }
                else
                {
/*                    cuda::SynapticImpactMessage message_out{
                        {projection.uid_, step_n},
                        projection.presynaptic_uid_,
                        projection.postsynaptic_uid_,
                        is_forcing<cuda::CUDAProjection<synapse_traits::DeltaSynapse>>(),
                        {impact}};

                    projection.messages_.push_back(std::make_pair(future_step, message_out));
*/
                }
            }
        }
    }

/*
    // WeightUpdateSTDP<SynapseType>::modify_weights(projection);
    return future_messages.find(step_n);
    //

    if (out_iter != future_messages.end())
    {
        // Send a message and remove it from the queue.
        message_bus_.send_message(out_iter->second);
        future_messages.erase(out_iter);
    }
*/
}


__device__ void CUDABackendImpl::calculate_projection(
    CUDAProjection<knp::synapse_traits::AdditiveSTDPDeltaSynapse> &projection,
    knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::SpikeMessage> &messages,
    std::uint64_t step_n)
{
    //SPDLOG_TRACE("Calculate AdditiveSTDPDelta synapse projection {}.", std::string(projection.get_uid()));
}


__device__ void CUDABackendImpl::calculate_projection(
    CUDAProjection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse> &projection,
    knp::backends::gpu::cuda::device_lib::CUDAVector<cuda::SpikeMessage> &messages,
    std::uint64_t step_n)
{
//    SPDLOG_TRACE("Calculate STDPSynapticResource synapse projection {}.", std::string(projection.get_uid()));
}


__host__ __device__ CUDABackendImpl::PopulationIterator CUDABackendImpl::begin_populations()
{
    return PopulationIterator{device_populations_.begin()};
}


__host__ __device__ CUDABackendImpl::PopulationConstIterator CUDABackendImpl::begin_populations() const
{
    return {device_populations_.cbegin()};
}


__host__ __device__ CUDABackendImpl::PopulationIterator CUDABackendImpl::end_populations()
{
    return PopulationIterator{device_populations_.end()};
}


__host__ __device__ CUDABackendImpl::PopulationConstIterator CUDABackendImpl::end_populations() const
{
    return device_populations_.cend();
}


__host__ __device__ CUDABackendImpl::ProjectionIterator CUDABackendImpl::begin_projections()
{
    return ProjectionIterator{device_projections_.begin()};
}


__host__ __device__ CUDABackendImpl::ProjectionConstIterator CUDABackendImpl::begin_projections() const
{
    return device_projections_.cbegin();
}


__host__ __device__ CUDABackendImpl::ProjectionIterator CUDABackendImpl::end_projections()
{
    return ProjectionIterator{device_projections_.end()};
}


__host__ __device__ CUDABackendImpl::ProjectionConstIterator CUDABackendImpl::end_projections() const
{
    return device_projections_.cend();
}

}  // namespace knp::backends::gpu::cuda

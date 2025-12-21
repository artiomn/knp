/**
 * @file delta_synapse_projection.h
 * @brief Common templates for delta synapse projection.
 * @kaspersky_support A. Vartenkov
 * @date 07.11.2023
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
#include <knp/backends/cpu-library/impl/delta_synapse_projection_impl.h>

#include <unordered_map>
/**
 * @brief Namespace for CPU backends.
 */
namespace knp::backends::cpu
{
/**
 * @brief Make one execution step for a projection of delta synapses.
 * @tparam DeltaLikeSynapseType type of a synapse that requires synapse weight and delay as parameters.
 * @param projection projection to update.
 * @param endpoint message endpoint used for message exchange.
 * @param future_messages message queue to process via endpoint.
 * @param step_n execution step.
 */
template <class DeltaLikeSynapseType>
void calculate_delta_synapse_projection(
    knp::core::Projection<DeltaLikeSynapseType> &projection, knp::core::MessageEndpoint &endpoint,
    MessageQueue &future_messages, size_t step_n)
{
    calculate_delta_synapse_projection_impl<DeltaLikeSynapseType>(projection, endpoint, future_messages, step_n);
}


/**
 * @brief Process a part of projection synapses.
 * @tparam DeltaLikeSynapse type of a synapse that requires synapse weight and delay as parameters.
 * @param projection projection to receive the message.
 * @param message_in_data processed spike data for the projection.
 * @param future_messages queue of future messages.
 * @param step_n current step.
 * @param part_start index of the starting synapse.
 * @param part_size number of synapses to process.
 * @param mutex mutex.
 */
template <class DeltaLikeSynapse>
void calculate_projection_part(
    knp::core::Projection<DeltaLikeSynapse> &projection, const std::unordered_map<knp::core::Step, size_t> &message_in_data,
    MessageQueue &future_messages, uint64_t step_n, size_t part_start, size_t part_size, std::mutex &mutex)
{
    calculate_projection_part_impl(projection, message_in_data, future_messages, step_n, part_start, part_size, mutex);
}

}  // namespace knp::backends::cpu

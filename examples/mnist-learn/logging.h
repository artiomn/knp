/**
 * @file logging.h
 * @brief Functions for network construction.
 * @kaspersky_support A. Vartenkov
 * @date 24.03.2025
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
#include <knp/core/messaging/messaging.h>
#include <knp/core/uid.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/monitoring/observer.h>

#include <fstream>
#include <map>
#include <string>
#include <vector>

/**
 * @brief Function object that receives messages and processes their data somehow.
 */
using SpikeProcessor = knp::framework::monitoring::MessageProcessor<knp::core::messaging::SpikeMessage>;


/**
 * @brief Structure that stores inference results from a single population.
 */
struct InferenceResult
{
    /**
     * @brief Response creation step.
     */
    size_t step_ = 0;

    /**
     * @brief Indexes of spiking neurons.
     */
    std::vector<int> indexes_{};
};

/**
 * @brief Create a header for aggregated log file.
 */
void write_aggregated_log_header(std::ofstream &log_stream, const std::map<knp::core::UID, std::string> &pop_names);


/**
 * @brief Create an observer function that outputs resulting spikes to terminal.
 * @param result spikes to be printed.
 * @return Function that outputs the spikes it receives.
 */
SpikeProcessor make_observer_function(std::vector<InferenceResult> &result);


/**
 * @brief Create a function that writes projection weights and update times to file.
 * @param weights_log output file stream.
 * @param period how often to write: the weights are only saved if a step is N * period.
 * @param model_executor model executor.
 * @param uid projection UID.
 * @return Function that writes projection weights to output stream.
 * @note Can only be used with Resource Delta Synapse projections.
 */
SpikeProcessor make_projection_observer_function(
    std::ofstream &weights_log, size_t period, knp::framework::ModelExecutor &model_executor,
    const knp::core::UID &uid);


/**
 * @brief Add a logger that calculates all spikes from a projection and writes them to a csv file.
 * @param model network model.
 * @param all_senders_names sender UID-name correspondence.
 * @param model_executor model executor.
 * @param current_index parameter to save integer state between runs.
 * @param spike_accumulator parameter to save accumulated spikes between runs.
 * @param log_stream output file stream.
 * @param aggregation_period aggregation period.
 */
void add_aggregate_logger(
    const knp::framework::Model &model, const std::map<knp::core::UID, std::string> &all_senders_names,
    knp::framework::ModelExecutor &model_executor, size_t &current_index,
    std::map<std::string, size_t> &spike_accumulator, std::ofstream &log_stream, int aggregation_period);

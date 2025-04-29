/**
 * @file logging.cpp
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

#include "logging.h"

#include <knp/core/messaging/messaging.h>
#include <knp/core/projection.h>
#include <knp/framework/model_executor.h>
#include <knp/synapse-traits/stdp_synaptic_resource_rule.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <tuple>


using ResourceDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;


SpikeProcessor make_observer_function(std::vector<InferenceResult> &result)
{
    auto observer_func = [&result](const std::vector<knp::core::messaging::SpikeMessage> &messages)
    {
        if (messages.empty() || messages[0].neuron_indexes_.empty()) return;
        InferenceResult result_buf;
        result_buf.step_ = messages[0].header_.send_time_;
        for (auto index : messages[0].neuron_indexes_)
        {
            std::cout << index << " ";
            result_buf.indexes_.push_back(index);
        }
        result.push_back(result_buf);
        std::cout << std::endl;
    };
    return observer_func;
}


auto fill_projection_weights(const knp::core::AllProjectionsVariant &proj_variant)
{
    const auto &proj = std::get<ResourceDeltaProjection>(proj_variant);
    std::vector<std::tuple<int, int, float, knp::core::Step>> weights_by_receiver_sender;
    for (const auto &synapse_data : proj)
    {
        float weight = std::get<0>(synapse_data).rule_.synaptic_resource_;
        knp::core::Step update_step = std::get<0>(synapse_data).rule_.last_spike_step_;
        size_t sender = std::get<1>(synapse_data);
        size_t receiver = std::get<2>(synapse_data);
        weights_by_receiver_sender.push_back({receiver, sender, weight, update_step});
    }
    std::sort(
        weights_by_receiver_sender.begin(), weights_by_receiver_sender.end(),
        [](const auto &v1, const auto &v2)
        {
            if (std::get<0>(v1) != std::get<0>(v2)) return std::get<0>(v1) < std::get<0>(v2);
            return std::get<1>(v1) < std::get<1>(v2);
        });
    return weights_by_receiver_sender;
}


SpikeProcessor make_projection_observer_function(
    std::ofstream &weights_log, size_t period, knp::framework::ModelExecutor &model_executor, const knp::core::UID &uid)
{
    auto observer_func =
        [&weights_log, period, &model_executor, uid](const std::vector<knp::core::messaging::SpikeMessage> &)
    {
        size_t step = model_executor.get_backend()->get_step();
        if (!weights_log.good() || step % period != 0) return;
        // Output weights for every step that is a full square
        weights_log << "Step: " << step << std::endl;
        const auto ranges = model_executor.get_backend()->get_network_data();
        for (auto &iter = *ranges.projection_range.first; iter != *ranges.projection_range.second; ++iter)
        {
            const knp::core::UID curr_proj_uid = std::visit([](const auto &proj) { return proj.get_uid(); }, *iter);
            if (curr_proj_uid != uid) continue;
            auto weights_by_receiver_sender = fill_projection_weights(*iter);
            size_t neuron = -1;
            for (const auto &syn_data : weights_by_receiver_sender)
            {
                size_t new_neuron = std::get<0>(syn_data);
                if (neuron != new_neuron)
                {
                    neuron = new_neuron;
                    weights_log << std::endl << "Neuron " << neuron << std::endl;
                }
                weights_log << std::get<2>(syn_data) << "|" << std::get<3>(syn_data) << " ";
            }
            weights_log << std::endl;
            return;
        }
    };
    return observer_func;
}


void write_aggregated_log_header(std::ofstream &log_stream, const std::map<knp::core::UID, std::string> &pop_names)
{
    std::vector<std::string> vec(pop_names.size());
    std::transform(pop_names.begin(), pop_names.end(), vec.begin(), [](const auto &val) { return val.second; });
    std::sort(vec.begin(), vec.end());
    log_stream << "Index";
    for (const auto &name : vec) log_stream << ", " << name;
    log_stream << std::endl;
}


void save_aggregated_log(std::ofstream &log_stream, const std::map<std::string, size_t> &values, size_t index)
{
    // Write values in order. Map is sorted by key value, that means by population name.
    log_stream << index;
    for (const auto &name_count_pair : values)
    {
        log_stream << ", " << name_count_pair.second;
    }
    log_stream << std::endl;
}


SpikeProcessor make_aggregate_observer(
    std::ofstream &log_stream, int period, const std::map<knp::core::UID, std::string> &pop_names,
    std::map<std::string, size_t> &accumulator, size_t &curr_index)
{
    // Initialize accumulator
    accumulator.clear();
    for (const auto &val : pop_names) accumulator.insert({val.second, 0});
    curr_index = 0;

    auto observer_func = [&log_stream, &accumulator, pop_names, period,
                          &curr_index](const std::vector<knp::core::messaging::SpikeMessage> &messages)
    {
        if (curr_index != 0 && curr_index % period == 0)
        {
            // Write container to log
            save_aggregated_log(log_stream, accumulator, curr_index);
            // Reset container
            accumulator.clear();
            for (const auto &val : pop_names) accumulator.insert({val.second, 0});
        }

        // Add spike numbers to accumulator
        for (const auto &msg : messages)
        {
            auto name_iter = pop_names.find(msg.header_.sender_uid_);
            if (name_iter == pop_names.end()) continue;
            std::string population_name = name_iter->second;
            accumulator[population_name] += msg.neuron_indexes_.size();
        }
        ++curr_index;
    };
    return observer_func;
}


std::string get_time_string()
{
    auto time_now = std::chrono::system_clock::now();
    std::time_t c_time = std::chrono::system_clock::to_time_t(time_now);
    std::string result(std::ctime(&c_time));
    return result;
}


void add_aggregate_logger(
    const knp::framework::Model &model, const std::map<knp::core::UID, std::string> &all_senders_names,
    knp::framework::ModelExecutor &model_executor, size_t &current_index,
    std::map<std::string, size_t> &spike_accumulator, std::ofstream &log_stream, int aggregation_period)
{
    std::vector<knp::core::UID> all_senders_uids;
    for (const auto &pop : model.get_network().get_populations())
    {
        knp::core::UID pop_uid = std::visit([](const auto &p) { return p.get_uid(); }, pop);
        all_senders_uids.push_back(pop_uid);
    }
    write_aggregated_log_header(log_stream, all_senders_names);
    model_executor.add_observer<knp::core::messaging::SpikeMessage>(
        make_aggregate_observer(log_stream, aggregation_period, all_senders_names, spike_accumulator, current_index),
        all_senders_uids);
}

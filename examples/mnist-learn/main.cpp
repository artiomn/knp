/**
 * @file main.cpp
 * @brief Example of training a MNIST network
 * @kaspersky_support A. Vartenkov
 * @date 30.08.2024
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

#include <knp/core/projection.h>
#include <knp/framework/model.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/monitoring/observer.h>
#include <knp/framework/network.h>
#include <knp/framework/sonata/network_io.h>
#include <knp/synapse-traits/all_traits.h>

#include <chrono>
#include <ctime>
#include <filesystem>
#include <functional>
#include <iostream>
#include <optional>

#include "construct_network.h"
#include "data_read.h"
#include "evaluation.h"
#include "logging.h"

namespace fs = std::filesystem;

using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using ResourceDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;

constexpr int num_subnetworks = 1;
constexpr int learning_period = 200000;
constexpr int testing_period = 10000;
constexpr int logging_aggregation_period = 4000;
constexpr int logging_weights_period = 100000;

// Create a spike message generator from an array of boolean frames.
auto make_input_generator(const std::vector<std::vector<bool>> &spike_frames, int64_t offset)
{
    auto generator = [&spike_frames, offset](knp::core::Step step)
    {
        knp::core::messaging::SpikeData message;
        if ((step + offset) >= spike_frames.size()) return message;

        for (size_t i = 0; i < spike_frames[step + offset].size(); ++i)
        {
            if (spike_frames[step + offset][i]) message.push_back(i);
        }
        return message;
    };

    return generator;
}


knp::framework::Network get_network_for_inference(
    const knp::core::Backend &backend, const std::set<knp::core::UID> &inference_population_uids,
    const std::set<knp::core::UID> &inference_internal_projection)
{
    auto data_ranges = backend.get_network_data();
    knp::framework::Network res_network;
    for (auto &iter = *data_ranges.population_range.first; iter != *data_ranges.population_range.second; ++iter)
    {
        auto population = *iter;
        knp::core::UID pop_uid = std::visit([](const auto &p) { return p.get_uid(); }, population);
        if (inference_population_uids.find(pop_uid) != inference_population_uids.end())
            res_network.add_population(std::move(population));
    }
    for (auto &iter = *data_ranges.projection_range.first; iter != *data_ranges.projection_range.second; ++iter)
    {
        auto projection = *iter;
        knp::core::UID proj_uid = std::visit([](const auto &p) { return p.get_uid(); }, projection);
        if (inference_internal_projection.find(proj_uid) != inference_internal_projection.end())
            res_network.add_projection(std::move(projection));
    }
    return res_network;
}


std::string get_time_string()
{
    auto time_now = std::chrono::system_clock::now();
    std::time_t c_time = std::chrono::system_clock::to_time_t(time_now);
    std::string result(std::ctime(&c_time));
    return result;
}


std::vector<knp::core::UID> add_wta_handlers(const AnnotatedNetwork &network, knp::framework::ModelExecutor &executor)
{
    std::vector<size_t> borders;
    std::vector<knp::core::UID> result;
    for (size_t i = 0; i < 10; ++i) borders.push_back(15 * i);
    // std::random_device rnd_device;
    int seed = 0;  // rnd_device();
    std::cout << "Seed " << seed << std::endl;
    for (const auto &senders_receivers : network.data_.wta_data)
    {
        knp::core::UID handler_uid;
        executor.add_spike_message_handler(
            knp::framework::modifier::KWtaPerGroup{borders, 1, seed++}, senders_receivers.first,
            senders_receivers.second, handler_uid);
        result.push_back(handler_uid);
    }
    return result;
}


AnnotatedNetwork train_mnist_network(
    const fs::path &path_to_backend, const std::vector<std::vector<bool>> &spike_frames,
    const std::vector<std::vector<bool>> &spike_classes, const fs::path &log_path = "")
{
    AnnotatedNetwork example_network = create_example_network_new(num_subnetworks);
    std::filesystem::create_directory("mnist_network");
    knp::framework::sonata::save_network(example_network.network_, "mnist_network");
    knp::framework::Model model(std::move(example_network.network_));

    knp::core::UID input_image_channel_raster;
    knp::core::UID input_image_channel_classses;

    for (auto image_proj_uid : example_network.data_.projections_from_raster)
        model.add_input_channel(input_image_channel_raster, image_proj_uid);
    for (auto target_proj_uid : example_network.data_.projections_from_classes)
        model.add_input_channel(input_image_channel_classses, target_proj_uid);

    knp::framework::ModelLoader::InputChannelMap channel_map;

    channel_map.insert({input_image_channel_raster, make_input_generator(spike_frames, 0)});  // -4 was better
    channel_map.insert({input_image_channel_classses, make_input_generator(spike_classes, 0)});

    knp::framework::BackendLoader backend_loader;
    knp::framework::ModelExecutor model_executor(model, backend_loader.load(path_to_backend), std::move(channel_map));
    std::vector<InferenceResult> result;

    // Add observer.
    model_executor.add_observer<knp::core::messaging::SpikeMessage>(
        make_observer_function(result), example_network.data_.output_uids);

    // Add all spikes observer.
    // These variables should have the same lifetime as model_executor, or else UB.
    std::ofstream log_stream, weight_stream, all_spikes_stream;
    std::map<std::string, size_t> spike_accumulator;
    // cppcheck-suppress variableScope
    size_t current_index = 0;
    std::vector<knp::core::UID> wta_uids = add_wta_handlers(example_network, model_executor);
    // All loggers go here
    if (!log_path.empty())
    {
        std::vector<knp::core::UID> all_populations_uids;
        for (const auto &pop : model.get_network().get_populations())
        {
            knp::core::UID pop_uid = std::visit([](const auto &p) { return p.get_uid(); }, pop);
            all_populations_uids.push_back(pop_uid);
        }
        log_stream.open(log_path / "spikes_training.csv", std::ofstream::out);
        auto all_names = example_network.data_.population_names;
        for (const auto &uid : wta_uids) all_names.insert({uid, "WTA"});

        if (log_stream.is_open())
        {
            write_aggregated_log_header(log_stream, all_names);
            model_executor.add_observer<knp::core::messaging::SpikeMessage>(
                make_aggregate_observer(
                    log_stream, logging_aggregation_period, example_network.data_.population_names, spike_accumulator,
                    current_index),
                all_populations_uids);
        }
        else
            std::cout << "Couldn't open log file at " << log_path << std::endl;
        weight_stream.open(log_path / "weights.log", std::ofstream::out);
        if (weight_stream.is_open())
        {
            model_executor.add_observer<knp::core::messaging::SpikeMessage>(
                make_projection_observer_function(
                    weight_stream, logging_weights_period, model_executor,
                    example_network.data_.projections_from_raster[0]),
                {});
        }
    }

    // Start model.
    std::cout << get_time_string() << ": learning started\n";

    model_executor.start(
        [](size_t step)
        {
            if (step % 20 == 0) std::cout << "Step: " << step << std::endl;
            return step != learning_period;
        });

    std::cout << get_time_string() << ": learning finished\n";
    example_network.network_ = get_network_for_inference(
        *model_executor.get_backend(), example_network.data_.inference_population_uids,
        example_network.data_.inference_internal_projection);
    return example_network;
}


std::vector<knp::core::messaging::SpikeMessage> run_mnist_inference(
    const fs::path &path_to_backend, AnnotatedNetwork &described_network,
    const std::vector<std::vector<bool>> &spike_frames, const fs::path &log_path = "")
{
    knp::framework::BackendLoader backend_loader;
    knp::framework::Model model(std::move(described_network.network_));

    // Creates arbitrary o_channel_uid identifier for the output channel.
    knp::core::UID o_channel_uid;
    // Passes to the model object the created output channel ID (o_channel_uid)
    // and the population IDs.
    knp::framework::ModelLoader::InputChannelMap channel_map;
    knp::core::UID input_image_channel_uid;
    channel_map.insert({input_image_channel_uid, make_input_generator(spike_frames, learning_period)});

    for (auto i : described_network.data_.output_uids) model.add_output_channel(o_channel_uid, i);
    for (auto image_proj_uid : described_network.data_.projections_from_raster)
        model.add_input_channel(input_image_channel_uid, image_proj_uid);
    knp::framework::ModelExecutor model_executor(model, backend_loader.load(path_to_backend), std::move(channel_map));

    // Receives a link to the output channel object (out_channel) from
    // the model executor (model_executor) by the output channel ID (o_channel_uid).
    auto &out_channel = model_executor.get_loader().get_output_channel(o_channel_uid);

    model_executor.get_backend()->stop_learning();
    std::vector<InferenceResult> result;

    // Add observer.
    model_executor.add_observer<knp::core::messaging::SpikeMessage>(
        make_observer_function(result), described_network.data_.output_uids);
    std::ofstream log_stream;
    // These two variables should have the same lifetime as model_executor, or else UB.
    std::map<std::string, size_t> spike_accumulator;
    // cppcheck-suppress variableScope
    size_t current_index = 0;
    auto wta_uids = add_wta_handlers(described_network, model_executor);
    auto all_senders_names = described_network.data_.population_names;
    for (const auto &uid : wta_uids)
    {
        all_senders_names.insert({uid, "WTA"});
    }

    // All loggers go here
    if (!log_path.empty())
    {
        log_stream.open(log_path / "spikes_inference.csv", std::ofstream::out);
        if (!log_stream.is_open()) std::cout << "Couldn't open log file : " << log_path << std::endl;
    }
    if (log_stream.is_open())
    {
        std::vector<knp::core::UID> all_senders_uids;
        for (const auto &pop : model.get_network().get_populations())
        {
            knp::core::UID pop_uid = std::visit([](const auto &p) { return p.get_uid(); }, pop);
            all_senders_uids.push_back(pop_uid);
        }
        write_aggregated_log_header(log_stream, all_senders_names);
        model_executor.add_observer<knp::core::messaging::SpikeMessage>(
            make_aggregate_observer(
                log_stream, logging_aggregation_period, described_network.data_.population_names, spike_accumulator,
                current_index),
            all_senders_uids);
    }

    // Start model.
    std::cout << get_time_string() << ": inference started\n";
    model_executor.start(
        [&spike_frames](size_t step)
        {
            if (step % 20 == 0) std::cout << "Inference step: " << step << std::endl;
            return step != testing_period;
        });
    // Creates the results vector that contains the indices of the spike steps.
    std::vector<knp::core::Step> results;
    // Updates the output channel.
    auto spikes = out_channel.update();
    std::sort(
        spikes.begin(), spikes.end(),
        [](const auto &sm1, const auto &sm2) { return sm1.header_.send_time_ < sm2.header_.send_time_; });
    return spikes;
}


int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Not enough parameters to run script: paths to both frames and targets are required" << std::endl;
        return EXIT_FAILURE;
    }
    fs::path log_path;
    if (argc >= 4) log_path = argv[3];

    // Defines path to backend, on which to run a network.
    std::filesystem::path path_to_backend =
        std::filesystem::path(argv[0]).parent_path() / "knp-cpu-single-threaded-backend";
    auto spike_frames = read_spike_frames(argv[1]);
    auto labels = read_labels(argv[2], learning_period);
    AnnotatedNetwork trained_network = train_mnist_network(path_to_backend, spike_frames, labels.train_, log_path);

    auto spikes = run_mnist_inference(path_to_backend, trained_network, spike_frames, log_path);
    std::cout << get_time_string() << ": inference finished  -- output spike count is " << spikes.size() << std::endl;

    process_inference_results(spikes, labels.test_, testing_period);
    return EXIT_SUCCESS;
}

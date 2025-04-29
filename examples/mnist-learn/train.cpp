/**
 * @file train.cpp
 * @brief Functions for train network.
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

#include "train.h"

#include <knp/framework/model.h>
#include <knp/framework/model_executor.h>
#include <knp/framework/monitoring/observer.h>
#include <knp/framework/network.h>
#include <knp/framework/sonata/network_io.h>
#include <knp/synapse-traits/all_traits.h>

#include <filesystem>
#include <map>
#include <utility>

#include "construct_network.h"
#include "data_read.h"
#include "inference.h"
#include "logging.h"
#include "wta.h"


namespace fs = std::filesystem;


// Create channel map for training.
auto build_channel_map_train(
    const AnnotatedNetwork &network, knp::framework::Model &model, const std::vector<std::vector<bool>> &spike_frames,
    const std::vector<std::vector<bool>> &spike_classes)
{
    // Create future channels uids randomly.
    knp::core::UID input_image_channel_raster;
    knp::core::UID input_image_channel_classses;

    // Add input channel for each image input projection.
    for (auto image_proj_uid : network.data_.projections_from_raster_)
        model.add_input_channel(input_image_channel_raster, image_proj_uid);

    // Add input channel for data labels.
    for (auto target_proj_uid : network.data_.projections_from_classes_)
        model.add_input_channel(input_image_channel_classses, target_proj_uid);

    // Create and fill a channel map.
    knp::framework::ModelLoader::InputChannelMap channel_map;
    channel_map.insert({input_image_channel_raster, make_input_generator(spike_frames, 0)});
    channel_map.insert({input_image_channel_classses, make_input_generator(spike_classes, 0)});

    return channel_map;
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


AnnotatedNetwork train_mnist_network(
    const fs::path &path_to_backend, const std::vector<std::vector<bool>> &spike_frames,
    const std::vector<std::vector<bool>> &spike_classes, const fs::path &log_path)
{
    AnnotatedNetwork example_network = create_example_network(num_subnetworks);
    std::filesystem::create_directory("mnist_network");
    knp::framework::sonata::save_network(example_network.network_, "mnist_network");
    knp::framework::Model model(std::move(example_network.network_));

    knp::framework::ModelLoader::InputChannelMap channel_map =
        build_channel_map_train(example_network, model, spike_frames, spike_classes);

    knp::framework::BackendLoader backend_loader;
    knp::framework::ModelExecutor model_executor(model, backend_loader.load(path_to_backend), std::move(channel_map));
    std::vector<InferenceResult> result;

    // Add observer.
    model_executor.add_observer<knp::core::messaging::SpikeMessage>(
        make_observer_function(result), example_network.data_.output_uids_);

    // Add all spikes observer.
    // These variables should have the same lifetime as model_executor, or else UB.
    std::ofstream log_stream, weight_stream;
    std::map<std::string, size_t> spike_accumulator;
    // cppcheck-suppress variableScope
    size_t current_index = 0;
    add_wta_handlers(example_network, model_executor);
    // All loggers go here
    if (!log_path.empty())
    {
        log_stream.open(log_path / "spikes_training.csv", std::ofstream::out);
        if (log_stream.is_open())
            add_aggregate_logger(
                model, example_network.data_.population_names_, model_executor, current_index, spike_accumulator,
                log_stream);
        else
            std::cout << "Couldn't open log file at " << log_path << std::endl;

        weight_stream.open(log_path / "weights.log", std::ofstream::out);
        if (weight_stream.is_open())
        {
            model_executor.add_observer<knp::core::messaging::SpikeMessage>(
                make_projection_observer_function(
                    weight_stream, logging_weights_period, model_executor,
                    example_network.data_.projections_from_raster_[0]),
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
        *model_executor.get_backend(), example_network.data_.inference_population_uids_,
        example_network.data_.inference_internal_projection_);
    return example_network;
}

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

#include <knp/framework/data_processing/image_classification.h>
#include <knp/framework/inference_evaluation/classification.h>

#include <filesystem>
#include <fstream>
#include <iostream>

#include "inference.h"
#include "time_string.h"
#include "train.h"

constexpr size_t active_steps = 10;
constexpr size_t steps_per_image = 20;
constexpr size_t image_size = 28 * 28;
constexpr float state_increment_factor = 1.f / 255;
constexpr size_t images_amount_to_train = 10000;
constexpr float dataset_split = 0.8;
constexpr size_t classes_amount = 10;


int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Not enough parameters.\n First parameter: path to frames file.\n "
                     "Second parameter: path to labels file.\n Third parameter (optional) path to log output directory."
                  << std::endl;
        return EXIT_FAILURE;
    }
    std::filesystem::path log_path;
    if (argc >= 4) log_path = argv[3];

    // Defines path to backend, on which to run a network.
    std::filesystem::path path_to_backend =
        std::filesystem::path(argv[0]).parent_path() / "knp-cpu-multi-threaded-backend";

    std::ifstream images_stream(argv[1], std::ios::binary);
    std::ifstream labels_stream(argv[2], std::ios::in);

    knp::framework::data_processing::image_classification::Dataset dataset =
        knp::framework::data_processing::image_classification::process_data(
            images_stream, labels_stream, images_amount_to_train, dataset_split, image_size, steps_per_image,
            knp::framework::data_processing::image_classification::make_simple_image_to_spikes_converter(
                steps_per_image, active_steps, image_size, state_increment_factor,
                std::vector<float>(image_size, 0.f)));

    std::cout << "Processed dataset, training will last " << dataset.steps_required_for_training_
              << " steps, inference " << dataset.steps_required_for_inference_ << " steps" << std::endl;

    // Construct network and run training.
    AnnotatedNetwork trained_network = train_mnist_network(path_to_backend, dataset, log_path);

    // Run inference for the same network.
    auto spikes = run_mnist_inference(path_to_backend, trained_network, dataset, log_path);
    std::cout << get_time_string() << ": inference finished  -- output spike count is " << spikes.size() << std::endl;

    // Evaluate results.
    auto const& processed_inference_results =
        knp::framework::inference_evaluation::classification::process_inference_results(
            spikes, dataset, classes_amount, steps_per_image);

    knp::framework::inference_evaluation::classification::write_inference_results_to_stream_as_csv(
        std::cout, processed_inference_results);

    return EXIT_SUCCESS;
}

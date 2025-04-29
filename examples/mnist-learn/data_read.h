/**
 * @file data_read.h
 * @brief Reading from dataset.
 * @kaspersky_support A. Vartenkov
 * @date 06.12.2024
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

#include <knp/core/core.h>
#include <knp/core/messaging/messaging.h>

#include <filesystem>
#include <functional>
#include <vector>

constexpr int intensity_levels = 10;
constexpr int frames_per_image = 20;
constexpr size_t input_size = 28 * 28;
constexpr double state_increment_factor = 1. / 255;

/**
 * @brief Labels for testing and training data.
 */
struct Labels
{
    std::vector<int> test_;
    std::vector<std::vector<bool>> train_;
};


/**
 * @brief Read input images.
 * @param path_to_data path to coded input images file.
 * @return vector of spike-based frames.
 */
std::vector<std::vector<bool>> read_spike_frames(
    const std::filesystem::path &path_to_data, int num_intensity_levels = intensity_levels);


/**
 * @brief Read labels for training and testing.
 * @param classes_file file with labels.
 * @param learning_period period for training.
 * @param offset how many frames to skip.
 * @return labels for training and testing.
 */
Labels read_labels(const std::filesystem::path &classes_file, int learning_period, int offset = 0);


// Create a spike message generator from an array of boolean frames.
std::function<knp::core::messaging::SpikeData(knp::core::Step)> make_input_generator(
    const std::vector<std::vector<bool>> &spike_frames, int64_t offset);

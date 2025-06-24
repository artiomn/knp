/**
 * @file inference.h
 * @brief Functions for inference.
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

#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "construct_network.h"


/// How many steps to use for testing.
constexpr int testing_period = 10000;

/**
 * @brief Run inference on MNIST dataset.
 * @param path_to_backend path to backend.
 * @param described_network trained network with descriptions.
 * @param spike_frames images file.
 * @param log_path path to log folder.
 * @return output spikes.
 */
std::vector<knp::core::messaging::SpikeMessage> run_mnist_inference(
    const std::filesystem::path &path_to_backend, AnnotatedNetwork &described_network,
    const std::vector<std::vector<bool>> &spike_frames, const std::filesystem::path &log_path = "");

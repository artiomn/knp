/**
 * @file train.h
 * @brief Functions for network training.
 * @kaspersky_support A. Vartenkov
 * @date 28.03.2025
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

/// How many subnetworks to use.
constexpr int num_subnetworks = 5;


/**
 * @brief Create and train a network.
 * @param path_to_backend path to backend.
 * @param spike_frames images file.
 * @param spike_classes labels file.
 * @param log_path path to log folder.
 * @return trained network with added descriptions.
 * @note the returned network is configured for inference.
 */
AnnotatedNetwork train_mnist_network(
    const std::filesystem::path &path_to_backend, const std::vector<std::vector<bool>> &spike_frames,
    const std::vector<std::vector<bool>> &spike_classes, const std::filesystem::path &log_path = "");

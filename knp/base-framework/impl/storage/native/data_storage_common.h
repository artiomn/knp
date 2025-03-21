/**
 * @file data_storage_common.h
 * @brief Common functions and definitions for saving and loading data.
 * @kaspersky_support An. Vartenkov
 * @date 24.04.2024
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
#include <knp/core/messaging/messaging.h>

#include <array>
#include <vector>


namespace knp::framework::io::storage::native
{
constexpr int MAGIC_NUMBER = 2682;
constexpr std::array<int64_t, 2> VERSION{0, 1};


std::vector<knp::core::messaging::SpikeMessage> convert_node_time_arrays_to_messages(
    const std::vector<int64_t> &nodes, const std::vector<float> &timestamps, const knp::core::UID &uid,
    float time_per_step);
}  // namespace knp::framework::io::storage::native

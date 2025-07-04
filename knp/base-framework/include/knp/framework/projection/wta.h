/**
 * @file wta.h
 * @brief Functions for Winner Takes All.
 * @kaspersky_support D. Postnikov
 * @date 03.07.2025
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

#include <knp/framework/model_executor.h>
#include <knp/framework/network.h>

#include <utility>
#include <vector>

namespace knp::framework::projection
{

/**
 * @brief add wta(winner takes all) handlers to network
 * @param executor model executor
 * @param winners_amount amount of winners
 * @param wta_data for each compound network: a vector of senders and a vector of receivers
 */
std::vector<knp::core::UID> KNP_DECLSPEC add_wta_handlers(
    knp::framework::ModelExecutor& executor, size_t winners_amount,
    std::vector<std::pair<std::vector<knp::core::UID>, std::vector<knp::core::UID>>> const& wta_data);


}  //namespace knp::framework::projection

/**
 * @file wta.h
 * @brief Functions for Winner Takes All.
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

#include <knp/framework/model_executor.h>
#include <knp/framework/network.h>

#include <vector>

#include "construct_network.h"


std::vector<knp::core::UID> add_wta_handlers(const AnnotatedNetwork &network, knp::framework::ModelExecutor &executor);

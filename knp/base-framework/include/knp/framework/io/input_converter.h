/**
 * @file input_converter.h
 * @brief Header for input converter.
 * @kaspersky_support Vartenkov Andrey
 * @date 25.04.2023
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

#include <knp/core/impexp.h>
#include <knp/core/message_endpoint.h>
#include <knp/core/messaging/spike_message.h>
#include <knp/core/uid.h>

#include <functional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


/**
 * @brief Input channel namespace.
 */
namespace knp::framework::io::input
{
/**
 * @brief Functor used for generating input spike messages.
 */
using DataGenerator = std::function<core::messaging::SpikeData(core::Step)>;

}  // namespace knp::framework::io::input

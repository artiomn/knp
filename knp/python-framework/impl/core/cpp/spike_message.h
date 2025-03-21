/**
 * @file spike_message.h
 * @brief Python bindings header for SpikeMessage.
 * @kaspersky_support Artiom N.
 * @date 28.02.2024
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

#include <memory>
#include <utility>

#include "common.h"


std::shared_ptr<knp::core::messaging::SpikeMessage> spike_message_constructor(
    const py::tuple& header, const py::list& spikes)
{
    core::messaging::SpikeMessage sm = {
        {py::extract<core::UID>(header[0]), py::extract<size_t>(header[1])},
        py_iterable_to_vector<core::messaging::SpikeIndex>(spikes)};

    return std::make_shared<knp::core::messaging::SpikeMessage>(std::move(sm));
}

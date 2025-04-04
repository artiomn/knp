/**
 * @file spike_message.h
 * @brief Spike message class for CUDA.
 * @kaspersky_support Artiom N.
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

#include "message_header.cuh"


/**
 * @brief CUDA messaging namespace.
 */
namespace knp::backends::gpu::impl
{

/**
 * @brief Spike index type in the form of a 32-bit unsigned integer.
 */
using SpikeIndex = uint32_t;


/**
 * @brief List of spike indexes.
 */
using SpikeData = std::device_vector<SpikeIndex>;


/**
 * @brief Structure of the spike message.
 */
struct SpikeMessage
{
    /**
     * @brief Message header.
     */
    MessageHeader header_;

    /**
     * @brief Indexes of the recently spiked neurons.
     */
    SpikeData neuron_indexes_;
};

}  // namespace knp::backends::gpu::impl

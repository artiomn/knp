/**
 * @file spike_message.cpp
 * @brief Spike message implementation header.
 * @kaspersky_support Artiom N.
 * @date 13.04.2023
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

#include <knp/core/messaging/spike_message.h>

#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdocumentation"
#endif
#include <knp_gen_headers/spike_message_generated.h>
#ifdef __clang__
#    pragma clang diagnostic pop
#endif

/**
 * @brief Messaging namespace.
 */
namespace knp::core::messaging
{
::flatbuffers::uoffset_t pack_internal(::flatbuffers::FlatBufferBuilder &builder, const SpikeMessage &msg);
SpikeMessage unpack(const marshal::SpikeMessage *s_msg);
}  // namespace knp::core::messaging

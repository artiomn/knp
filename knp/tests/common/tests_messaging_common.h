/**
 * @file tests_messaging_common.h
 * @brief Common routines used for messaging in tests.
 * @kaspersky_support Artiom N.
 * @date 16.10.2025
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

#include <knp/core/core.h>
#include <knp/core/messaging/messaging.h>

#include <spdlog/spdlog.h>

#include <vector>


namespace testing::internal
{

template <class Endpoint>
bool send_messages_smallest_network(const knp::core::UID &in_channel_uid, Endpoint &endpoint, knp::core::Step step)
{
    if (step % 5 == 0)
    {
        knp::core::messaging::SpikeMessage message{{in_channel_uid, 0}, {0}};
        endpoint.send_message(message);
        return true;
    }
    return false;
}


template <class Endpoint>
bool receive_messages_smallest_network(const knp::core::UID &out_channel_uid, Endpoint &endpoint)
{
    endpoint.receive_all_messages();
    // Write the steps on which the network sends a spike.
    if (!endpoint.template unload_messages<knp::core::messaging::SpikeMessage>(out_channel_uid).empty()) return true;
    return false;
}

}  //namespace testing::internal

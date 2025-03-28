/**
 * @file wta.cpp
 * @brief Functions for Winner Takes All implementation.
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

#include "wta.h"

#include <knp/framework/monitoring/observer.h>
#include <knp/framework/sonata/network_io.h>
#include <knp/synapse-traits/all_traits.h>

#include <utility>

std::vector<knp::core::UID> add_wta_handlers(const AnnotatedNetwork &network, knp::framework::ModelExecutor &executor)
{
    std::vector<size_t> borders;
    std::vector<knp::core::UID> result;

    for (size_t i = 0; i < 10; ++i) borders.push_back(15 * i);
    // std::random_device rnd_device;
    int seed = 0;  // rnd_device();
    std::cout << "Seed " << seed << std::endl;
    for (const auto &senders_receivers : network.data_.wta_data_)
    {
        knp::core::UID handler_uid;
        executor.add_spike_message_handler(
            knp::framework::modifier::KWtaPerGroup{borders, 1, seed++}, senders_receivers.first,
            senders_receivers.second, handler_uid);
        result.push_back(handler_uid);
    }
    return result;
}

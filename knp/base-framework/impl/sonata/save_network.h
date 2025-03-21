/**
 * @file save_network.h
 * @brief Saving network to SONATA format file.
 * @kaspersky_support An. Vartenkov
 * @date 15.04.2024
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

#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "highfive.h"


namespace knp::framework::sonata
{
namespace fs = std::filesystem;

template <class Projection>
void add_projection_to_h5(HighFive::File &, const Projection &);


template <class Population>
void add_population_to_h5(HighFive::File &, const Population &);


// Read parameter values for both projections and populations.
template <class Attr>
std::vector<Attr> read_parameter(
    const HighFive::Group &population_group, const std::string &param_name, size_t pop_size, const Attr &default_value)
{
    std::vector<Attr> result(pop_size);
    try
    {
        auto dataset = population_group.getDataSet(param_name);
        dataset.read(result);
    }
    catch (std::exception &exc)
    {
        return std::vector(pop_size, default_value);
    }
    return result;
}

}  // namespace knp::framework::sonata


#define PUT_NEURON_TO_DATASET(pop, param, group)                                                                \
    do                                                                                                          \
    {                                                                                                           \
        std::vector<decltype(pop.begin()->param)> data;                                                         \
        data.reserve(pop.size());                                                                               \
        std::transform(                                                                                         \
            pop.begin(), pop.end(), std::back_inserter(data), [](const auto &neuron) { return neuron.param; }); \
        group.createDataSet(#param, data);                                                                      \
    } while (false)

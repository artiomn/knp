/**
 * @file common_procedures.h
 * @brief Type-independent procedures for saving and loading.
 * @kaspersky_support A. Vartenkov
 * @date 01.04.2025
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
#include <spdlog/spdlog.h>

#include <string>
#include <vector>

#include "../highfive.h"
#include "type_id_defines.h"


namespace knp::framework::sonata
{

const auto dynamic_subgroup_name = "dynamics_params";


template <class Population>
HighFive::Group initialize_adding_population(const Population &population, HighFive::File &file_h5)
{
    SPDLOG_TRACE("Adding population {} to HDF5...", std::string(population.get_uid()));

    // Check that an external function has created "nodes" group.
    if (!file_h5.exist("nodes"))
    {
        throw std::runtime_error("File does not contain the \"nodes\" group.");
    }

    HighFive::Group population_group = file_h5.createGroup("nodes/" + std::string{population.get_uid()});

    std::vector<size_t> neuron_ids;
    // std::vector<int> neuron_type_ids(population.size(), get_neuron_type_id<neuron_traits::BLIFATNeuron>());
    neuron_ids.reserve(population.size());
    for (size_t i = 0; i < population.size(); ++i) neuron_ids.push_back(i);

    population_group.createDataSet("node_id", neuron_ids);
    population_group.createDataSet("node_group_index", neuron_ids);
    population_group.createDataSet("node_group_id", std::vector<size_t>(population.size(), 0));

    population_group.createDataSet(
        "node_type_id",
        std::vector<size_t>(population.size(), get_neuron_type_id<typename Population::PopulationNeuronType>()));
    auto group0 = population_group.createGroup("0");
    return group0;
}

}  // namespace knp::framework::sonata

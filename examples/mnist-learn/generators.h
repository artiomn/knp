/**
 * @file generators.h
 * @brief Generators creators.
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

#include <knp/core/projection.h>

using DeltaSynapseData = knp::synapse_traits::synapse_parameters<knp::synapse_traits::DeltaSynapse>;
using DeltaProjection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;
using ResourceSynapse = knp::synapse_traits::SynapticResourceSTDPDeltaSynapse;
using ResourceDeltaProjection = knp::core::Projection<knp::synapse_traits::SynapticResourceSTDPDeltaSynapse>;
using ResourceSynapseData = ResourceDeltaProjection::Synapse;
using ResourceSynapseParams = knp::synapse_traits::synapse_parameters<ResourceSynapse>;
using ResourceSynapseGenerator = std::function<ResourceSynapseData(size_t)>;


// A dense projection generator from a default synapse.
ResourceSynapseGenerator make_dense_generator(size_t from_size, const ResourceSynapseParams &default_synapse)
{
    ResourceSynapseGenerator synapse_generator = [from_size, default_synapse](size_t index)
    {
        size_t from_index = index % from_size;
        size_t to_index = index / from_size;
        // If you need to have synapses with different parameters, change them here.
        return ResourceSynapseData{default_synapse, from_index, to_index};
    };
    return synapse_generator;
}


DeltaProjection::SynapseGenerator make_aligned_generator(
    size_t prepopulation_size, size_t postpopulation_size, const DeltaSynapseData &default_synapse)
{
    DeltaProjection::SynapseGenerator synapse_generator =
        [prepopulation_size, postpopulation_size, default_synapse](size_t index)
    {
        size_t from_index;
        size_t pack_size;
        size_t to_index;
        if (prepopulation_size >= postpopulation_size)
        {
            from_index = index;
            pack_size = prepopulation_size / postpopulation_size;
            to_index = index / pack_size;
        }
        else
        {
            to_index = index;
            pack_size = postpopulation_size / prepopulation_size;
            from_index = index / pack_size;
        }
        return DeltaProjection::Synapse{default_synapse, from_index, to_index};
    };
    return synapse_generator;
}


// This generator makes all-to-all projection without 1-to-1 element.
DeltaProjection::SynapseGenerator make_exclusive_generator(
    size_t population_size, const DeltaSynapseData &default_synapse)
{
    DeltaProjection::SynapseGenerator synapse_generator = [population_size, default_synapse](size_t index)
    {
        size_t from_index;
        size_t to_index;
        from_index = index / (population_size - 1);
        to_index = index % (population_size - 1);
        if (to_index >= from_index) ++to_index;
        return DeltaProjection::Synapse{default_synapse, from_index, to_index};
    };
    return synapse_generator;
}

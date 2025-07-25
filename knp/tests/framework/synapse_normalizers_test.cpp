/**
 * @file synapse_normalizers_test.cpp
 * @brief Tests for neuron parameters normalizers.
 * @kaspersky_support Artiom N.
 * @date 24.07.2025
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

#include <knp/framework/normalization/normalize_synapses.h>
#include <knp/framework/normalization/synapse_accessors.h>
#include <knp/framework/projection/creators.h>
#include <knp/synapse-traits/delta.h>

#include <tests_common.h>


TEST(SynapseNormalizers, NormalizeWeight)
{
    constexpr size_t src_pop_size = 3;
    constexpr size_t dest_pop_size = 3;

    auto proj = knp::framework::projection::creators::all_to_all<typename knp::synapse_traits::DeltaSynapse>(
        knp::core::UID(), knp::core::UID(), src_pop_size, dest_pop_size);

    auto wa = knp::framework::projection::WeightAccessor<knp::synapse_traits::DeltaSynapse>(
        IntervalRecalculator<float>(10, 20, 0, 1));

    knp::framework::normalize_synapses(proj, wa);
}

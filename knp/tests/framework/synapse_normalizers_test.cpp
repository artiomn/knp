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

#include <knp/framework/normalization/synapse_accessors.h>
#include <knp/framework/normalization/synapse_normalizer.h>
#include <knp/framework/projection/creators.h>
#include <knp/synapse-traits/delta.h>

#include <tests_common.h>


TEST(SynapseNormalizers, NormalizeProjectionsWeights)
{
    constexpr size_t src_pop_size = 3;
    constexpr size_t dest_pop_size = 3;

    using SynapseType = knp::synapse_traits::DeltaSynapse;
    using SynapseParametersType = knp::core::Projection<SynapseType>::SynapseParameters;

    auto proj = knp::framework::projection::creators::all_to_all<SynapseType>(
        knp::core::UID(), knp::core::UID(), src_pop_size, dest_pop_size,
        [](size_t prev_n, size_t) -> SynapseParametersType
        {
            SynapseParametersType sp;

            sp.weight_ = (prev_n + 1) * 10;
            return sp;
        });

    for (const auto& synapse : proj)
    {
        const auto params = std::get<knp::core::synapse_data>(synapse);

        SPDLOG_DEBUG("Old synapse weight: {}.", params.weight_);

        ASSERT_GE(params.weight_, 10);
    }

    knp::framework::normalization::normalize_synapses<SynapseType>(
        proj, knp::framework::normalization::WeightAccessor<SynapseType>(),
        knp::framework::normalization::WeightAccessor<SynapseType>(),
        knp::framework::normalization::Rescaler<float>(10, dest_pop_size * 10, 0, 1));

    for (const auto& synapse : proj)
    {
        const auto params = std::get<knp::core::synapse_data>(synapse);

        SPDLOG_DEBUG("New synapse weight: {}.", params.weight_);

        ASSERT_GE(params.weight_, 0);
        ASSERT_LE(params.weight_, 1);
    }
}


TEST(SynapseNormalizers, NormalizeNetworkWeights)
{
    constexpr size_t src_pop_size = 3;
    constexpr size_t dest_pop_size = 3;

    using SynapseType = knp::synapse_traits::DeltaSynapse;
    using SynapseParametersType = knp::core::Projection<SynapseType>::SynapseParameters;

    auto proj = knp::framework::projection::creators::all_to_all<SynapseType>(
        knp::core::UID(), knp::core::UID(), src_pop_size, dest_pop_size,
        [](size_t prev_n, size_t) -> SynapseParametersType
        {
            SynapseParametersType sp;

            sp.weight_ = (prev_n + 1) * 10;
            return sp;
        });

    for (const auto& synapse : proj)
    {
        const auto params = std::get<knp::core::synapse_data>(synapse);

        SPDLOG_DEBUG("Old synapse weight: {}.", params.weight_);

        ASSERT_GE(params.weight_, 10);
    }

    knp::framework::Network net;
    net.add_projection(std::move(proj));

    knp::framework::normalization::normalize_synapses<
        knp::framework::normalization::WeightAccessor, knp::framework::normalization::WeightAccessor>(
        net, knp::framework::normalization::Rescaler<float>(10, dest_pop_size * 10, 0, 1));

    for (const auto& n_proj : net.get_projections())
    {
        std::visit(
            [](const auto& prj)
            {
                for (const auto& synapse : prj)
                {
                    const auto params = std::get<knp::core::synapse_data>(synapse);

                    SPDLOG_DEBUG("New synapse weight: {}.", params.weight_);

                    ASSERT_GE(params.weight_, 0);
                    ASSERT_LE(params.weight_, 1);
                }
            },
            n_proj);
    }
}

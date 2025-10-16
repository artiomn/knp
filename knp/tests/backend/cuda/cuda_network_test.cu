/**
 * @file cuda_network_test.cu
 * @brief CUDA smallest test.
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

#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <generators.h>
#include <spdlog/spdlog.h>
#include <tests_messaging_common.h>
#include <tests_common.h>

#include <functional>
#include <iostream>
#include <vector>

#include <knp/backends/gpu-cuda/backend.h>


using Population = knp::backends::gpu::CUDABackend::PopulationVariants;
using Projection = knp::backends::gpu::CUDABackend::ProjectionVariants;


namespace knp::testing
{

TEST(CudaBackendSuite, SmallestNetwork)
{
    // Create a single-neuron neural network: input -> input_projection -> population <=> loop_projection.

    namespace kt = knp::testing;
    backends::gpu::CUDABackend backend;

    kt::BLIFATPopulation population{kt::neuron_generator, 1};
    Projection loop_projection =
        kt::DeltaProjection{population.get_uid(), population.get_uid(), kt::synapse_generator, 1};
    Projection input_projection =
        kt::DeltaProjection{knp::core::UID{false}, population.get_uid(), kt::input_projection_gen, 1};
    knp::core::UID input_uid = std::visit([](const auto &proj) { return proj.get_uid(); }, input_projection);

    backend.load_populations({population});
    backend.load_projections({input_projection, loop_projection});

    auto endpoint = backend.get_message_bus().create_endpoint();

    knp::core::UID in_channel_uid;
    knp::core::UID out_channel_uid;

    // Create input and output.
    backend.subscribe<knp::core::messaging::SpikeMessage>(input_uid, {in_channel_uid});
    endpoint.subscribe<knp::core::messaging::SpikeMessage>(out_channel_uid, {population.get_uid()});

    std::vector<knp::core::Step> results;

//    backend._init();

    for (knp::core::Step step = 0; step < 20; ++step)
    {
        // Send inputs on steps 0, 5, 10, 15.
        ::testing::internal::send_messages_smallest_network(in_channel_uid, endpoint, step);
        backend._step();
        if (::testing::internal::receive_messages_smallest_network(out_channel_uid, endpoint)) results.push_back(step);
    }

    // Spikes on steps "5n + 1" (input) and on "previous_spike_n + 6" (positive feedback loop).
    const std::vector<knp::core::Step> expected_results = {1, 6, 7, 11, 12, 13, 16, 17, 18, 19};
    ASSERT_EQ(results, expected_results);
}

}  // namespace knp::testing

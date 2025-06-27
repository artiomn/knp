/**
 * @file cuda_test.cpp
 * @brief CUDA backend test.
 * @kaspersky_support Artiom N.
 * @date 26.02.2025
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

#include <knp/backends/gpu-cuda/backend.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <generators.h>
#include <spdlog/spdlog.h>
#include <tests_common.h>

#include <functional>
#include <vector>


using Population = knp::backends::gpu::CUDABackend::PopulationVariants;
using Projection = knp::backends::gpu::CUDABackend::ProjectionVariants;


namespace knp::testing
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


TEST(CUDABackendSuite, CUDADevice)  // cppcheck-suppress syntaxError
{
    auto gpus = knp::devices::gpu::list_cuda_processors();
    for (const auto &gpu : gpus)
    {
        auto gpu_ptr = dynamic_cast<const knp::devices::gpu::CUDA *>(&gpu);
        SPDLOG_INFO("GPU name: {}, warp size = {}", gpu.get_name(), gpu_ptr->get_warp_size());
    }
}


TEST(CUDABackendSuite, SmallestNetwork)
{
    // Create a single-neuron neural network: input -> input_projection -> population <=> loop_projection.

    namespace kt = knp::testing;

    // Spikes on steps "5n + 1" (input) and on "previous_spike_n + 6" (positive feedback loop).
    // const std::vector<knp::core::Step> expected_results = {1, 6, 7, 11, 12, 13, 16, 17, 18, 19};
    // ASSERT_EQ(results, expected_results);
}


TEST(CUDABackendSuite, NeuronsGettingTest)
{
    // const knp::testing::MTestingBack backend;

    // auto s_neurons = backend.get_supported_neurons();

    // ASSERT_LE(s_neurons.size(), boost::mp11::mp_size<knp::neuron_traits::AllNeurons>());
    // ASSERT_EQ(s_neurons[0], "BLIFATNeuron");
}


TEST(CUDABackendSuite, SynapsesGettingTest)
{
    // const knp::testing::MTestingBack backend;

    // auto s_synapses = backend.get_supported_synapses();

    // ASSERT_LE(s_synapses.size(), boost::mp11::mp_size<knp::synapse_traits::AllSynapses>());
    // ASSERT_EQ(s_synapses[0], "DeltaSynapse");
}

}  // namespace knp::testing

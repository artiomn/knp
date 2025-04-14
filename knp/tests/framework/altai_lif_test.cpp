/**
 * @file backend_loader_test.cpp
 * @brief Backend loading testing.
 * @kaspersky_support Artiom N.
 * @date 17.03.2023
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


#include <knp/backends/cpu-single-threaded/backend.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>
#include <knp/framework/network.h>
#include <knp/neuron-traits/altai_lif.h>
#include <knp/synapse-traits/delta.h>

#include <tests_common.h>

using Synapse = knp::synapse_traits::DeltaSynapse;

namespace knp::testing
{

class TestingBackendST : public knp::backends::single_threaded_cpu::SingleThreadedCPUBackend
{
public:
    TestingBackendST() = default;
    void _init() override { knp::backends::single_threaded_cpu::SingleThreadedCPUBackend::_init(); }
};

using Population = knp::core::Population<knp::neuron_traits::AltAILIF>;
using Projection = knp::core::Projection<knp::synapse_traits::DeltaSynapse>;

}  // namespace knp::testing


std::vector<int16_t> run_altai_neuron(
    const knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF> &neuron, size_t num_steps)
{
    knp::core::Population<knp::neuron_traits::AltAILIF> population{[&neuron](size_t) { return neuron; }, 1};
    knp::testing::TestingBackendST backend;
    backend.load_populations({population});
    backend._init();
    auto &pop = *backend.begin_populations();
    std::vector<int16_t> result;
    auto &neuron_ref = *std::get<knp::core::Population<knp::neuron_traits::AltAILIF>>(pop).begin();
    for (size_t step = 0; step < num_steps; ++step)
    {
        result.push_back(static_cast<int16_t>(neuron_ref.potential_));
        backend._step();
    }
    return result;
}


TEST(AltAiSuite, NeuronPotentialLeakRev)
{
    constexpr int starting_potential = 100;
    constexpr int potential_leak = -25;
    constexpr size_t num_steps = 10;
    auto base_neuron = knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF>{};
    base_neuron.activation_threshold_ = 2 * starting_potential;  // We don't want activations in this test.
    base_neuron.potential_leak_ = potential_leak;
    base_neuron.potential_ = starting_potential;
    base_neuron.leak_rev_ = true;

    // Potentials should be: 100, 75, 50, 25, 0, -25, 0, -25, etc.
    auto results = run_altai_neuron(base_neuron, num_steps);
    const std::vector<int16_t> expected_results = {100, 75, 50, 25, 0, -25, 0, -25, 0, -25};
    ASSERT_EQ(results, expected_results);
}


TEST(AltAiSuite, NeuronPotentialLeakNoRev)
{
    constexpr int starting_potential = 100;
    constexpr int potential_leak = -25;
    constexpr size_t num_steps = 10;
    auto base_neuron = knp::neuron_traits::neuron_parameters<knp::neuron_traits::AltAILIF>{};
    // This is a leak test, so no activations right now, negative or otherwise.
    base_neuron.activation_threshold_ = 2 * starting_potential;
    base_neuron.negative_activation_threshold_ = -2 * starting_potential;
    base_neuron.potential_leak_ = potential_leak;
    base_neuron.potential_ = starting_potential;
    base_neuron.leak_rev_ = false;

    auto results = run_altai_neuron(base_neuron, num_steps);
    const std::vector<int16_t> expected_results = {100, 75, 50, 25, 0, -25, -50, -75, -100, -125};
    ASSERT_EQ(results, expected_results);
}


TEST(AltAiTestSuite, NeuronMechanicsTest)
{
    // // Create a single-neuron neural network: input -> input_projection -> population <=> loop_projection.
    // knp::testing::STestingBack backend;
    //
    // knp::testing::Population population{knp::testing::neuron_generator, 1};
    // knp::testing::Projection loop_projection =
    //         knp::testing::Projection {population.get_uid(), population.get_uid(), knp::testing::synapse_generator,
    //         1};
    // knp::testing::Projection input_projection = knp::testing::Projection {
    //         knp::core::UID{false}, population.get_uid(), knp::testing::input_projection_gen, 1};
    // knp::core::UID const input_uid = std::visit([](const auto &proj) { return proj.get_uid(); }, input_projection);
    //
    // backend.load_populations({population});
    // backend.load_projections({input_projection, loop_projection});
    //
    // backend._init();
    // auto endpoint = backend.get_message_bus().create_endpoint();
    //
    // const knp::core::UID in_channel_uid, out_channel_uid;
    //
    // // Create input and output.
    // backend.subscribe<knp::core::messaging::SpikeMessage>(input_uid, {in_channel_uid});
    // endpoint.subscribe<knp::core::messaging::SpikeMessage>(out_channel_uid, {population.get_uid()});
    //
    // std::vector<knp::core::Step> results;
    //
    // for (knp::core::Step step = 0; step < 20; ++step)
    // {
    //     // Send inputs on steps 0, 5, 10, 15.
    //     if (step % 5 == 0)
    //     {
    //         knp::core::messaging::SpikeMessage message{{in_channel_uid, step}, {0}};
    //         endpoint.send_message(message);
    //     }
    //     backend._step();
    //     endpoint.receive_all_messages();
    //     // Write the steps on which the network sends a spike.
    //     if (!endpoint.unload_messages<knp::core::messaging::SpikeMessage>(out_channel_uid).empty())
    //     {
    //         results.push_back(step);
    //     }
    // }
    //
    // // Spikes on steps "5n + 1" (input) and on "previous_spike_n + 6" (positive feedback loop).
    // const std::vector<knp::core::Step> expected_results = {1, 6, 7, 11, 12, 13, 16, 17, 18, 19};
    // ASSERT_EQ(results, expected_results);
}

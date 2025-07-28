/**
 * @file cuda_test.cu
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

// #include <knp/backends/gpu-cuda/backend.h>
#include <knp/core/message_bus.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <generators.h>
#include <spdlog/spdlog.h>
#include <tests_common.h>

#include <functional>
#include <vector>

#include "../../backends/gpu/cuda-backend/impl/cuda_lib/vector.cuh"
#include "../../backends/gpu/cuda-backend/impl/cuda_bus/message_bus.cuh"


// using Population = knp::backends::gpu::CUDABackend::PopulationVariants;
// using Projection = knp::backends::gpu::CUDABackend::ProjectionVariants;


namespace knp::testing
{


// struct MessageBusTandem
// {
//     MessageBusTandem() : cpu_(knp::core::MessageBus::construct_bus()), gpu_(cpu_.create_endpoint())
//     {}
//     knp::core::MessageBus cpu_;
//     knp::backends::gpu::cuda::CUDAMessageBus gpu_;
// };


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


__device__ void prn()
{
    printf("Hello from GPU! Thread ID: %d\n", threadIdx.x);

    knp::backends::gpu::cuda::device_lib::CudaVector<int> cv;

    cv.reserve(2);

    printf("cv.size() = %lu\n", cv.size());
    cv.push_back(1);
    printf("cv.size() = %lu, v = %d\n", cv.size(), cv[0]);
    auto v = cv.pop_back();
    printf("p1 = %d\n", v);
    printf("cv.size() = %lu\n", cv.size());
    cv.resize(10);

    for (int i = 0; i < cv.size(); ++i) printf("i0 = %d\n", cv[i]);
    for (int i = 0; i < cv.size(); ++i) cv[i] = i;
    for (int i = 0; i < cv.size(); ++i) printf("i1 = %d\n", cv[i]);
    cv.resize(5);
    for (int i = 0; i < cv.size(); ++i) printf("i2 = %d\n", cv[i]);
    cv.reserve(15);
    for (int i = 0; i < cv.size(); ++i) printf("i3 = %d\n", cv[i]);
}


__global__ void run_bus()
{
    prn();
}


/*TEST(CudaBackendSuite, CUDADevice)  // cppcheck-suppress syntaxError
{
    auto gpus = knp::devices::gpu::list_cuda_processors();
    for (const auto &gpu : gpus)
    {
        auto gpu_ptr = dynamic_cast<const knp::devices::gpu::CUDA *>(&gpu);
        SPDLOG_INFO(
            "GPU name: {}, warp size = {}, power = {}", gpu.get_name(), gpu_ptr->get_warp_size(), gpu.get_power());
    }
}*/


TEST(CudaBackendSuite, CudaUidConversionTest)
{
    knp::core::UID orig_uid;
    knp::backends::gpu::cuda::UID cuda_uid = knp::backends::gpu::cuda::to_gpu_uid(orig_uid);
    knp::core::UID restored_uid = knp::backends::gpu::cuda::to_cpu_uid(cuda_uid);
    ASSERT_EQ(orig_uid, restored_uid);
}


TEST(CudaBackendSuite, MessagesTest)
{
    using namespace knp::backends::gpu::cuda;
    SpikeMessage message_1;
    SynapticImpactMessage message_2;
    ASSERT_EQ(message_1.neuron_indexes_.size(), 0);
    ASSERT_EQ(message_2.impacts_.size(), 0);
}


TEST(CudaBackendSuite, CudaSubscription)
{
    using namespace knp::backends::gpu::cuda;
    Subscription<SpikeMessage> subscription;
    std::cout << sizeof(subscription); // TODO tmp
}


TEST(CudaBackendSuite, CudaBusSubscription)
{
    // using knp::backends::gpu::cuda::to_gpu_uid;
    // using knp::backends::gpu::cuda::device_lib::CudaVector;
    // using knp::backends::gpu::cuda::UID;
    // MessageBusTandem bus_pair;
    // knp::core::UID sender_1, sender_2, receiver_1, receiver_2;
    // CudaVector<UID> senders_1, senders_2;
    // senders_1.push_back(to_gpu_uid(sender_1));
    // senders_1.push_back(to_gpu_uid(sender_2));
    // bus_pair.gpu_.subscribe<knp::backends::gpu::cuda::SpikeMessage>(
    //         to_gpu_uid(receiver_1), senders_1);
    // senders_2.push_back(to_gpu_uid(sender_1));
    // bus_pair.gpu_.subscribe<knp::backends::gpu::cuda::SpikeMessage>(
    //     to_gpu_uid(receiver_2), senders_2);
    // ASSERT_EQ(bus_pair.gpu_.get_subscriptions().size(), 2);


    // const knp::backends::gpu::cuda::SubscriptionVariant &sub_v = bus_pair.gpu_.get_subscriptions()[0];
    // const auto &sub = ::cuda::std::get<knp::backends::gpu::cuda::Subscription<knp::backends::gpu::cuda::SpikeMessage>>(sub_v);
    // ASSERT_EQ(sub.get_senders().size(), 2);
}


TEST(CudaBackendSuite, SmallestNetwork)
{
    // Create a single-neuron neural network: input -> input_projection -> population <=> loop_projection.

    namespace kt = knp::testing;

    // Spikes on steps "5n + 1" (input) and on "previous_spike_n + 6" (positive feedback loop).
    // const std::vector<knp::core::Step> expected_results = {1, 6, 7, 11, 12, 13, 16, 17, 18, 19};
    // ASSERT_EQ(results, expected_results);
}


TEST(CudaBackendSuite, NeuronsGettingTest)
{
    // const knp::testing::MTestingBack backend;

    // auto s_neurons = backend.get_supported_neurons();

    // ASSERT_LE(s_neurons.size(), boost::mp11::mp_size<knp::neuron_traits::AllNeurons>());
    // ASSERT_EQ(s_neurons[0], "BLIFATNeuron");
}


TEST(CudaBackendSuite, SynapsesGettingTest)
{
    // const knp::testing::MTestingBack backend;

    // auto s_synapses = backend.get_supported_synapses();

    // ASSERT_LE(s_synapses.size(), boost::mp11::mp_size<knp::synapse_traits::AllSynapses>());
    // ASSERT_EQ(s_synapses[0], "DeltaSynapse");
}

}  // namespace knp::testing

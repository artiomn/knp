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
#include <iostream>
#include <vector>


#include "../../backends/gpu/cuda-backend/impl/cuda_lib/vector.cuh"
#include "../../backends/gpu/cuda-backend/impl/cuda_bus/message_bus.cuh"
#include "../../backends/gpu/cuda-backend/impl/cuda_bus/messaging.cuh"
#include "../../backends/gpu/cuda-backend/impl/uid.cuh"


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
    for (int i = 0; i < cv.size(); ++i) cv.set(i, i);
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

TEST(CudaContainerSuite, Memcpy)
{
    const uint64_t val = 112;
    uint64_t *val_gpu;
    uint64_t val_cpu = 0;
    cudaMalloc(&val_gpu, sizeof(uint64_t));
    cudaMemcpy(val_gpu, &val, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&val_cpu, val_gpu, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(val_gpu);
    ASSERT_EQ(val, val_cpu);
}


TEST(CudaContainerSuite, MemcpyArray)
{
    const cuda::std::array<uint64_t, 4> array{1, 2, 3, 4};
    cuda::std::array<uint64_t, 4> *array_gpu;
    cuda::std::array<uint64_t, 4> array_cpu{4, 3, 2, 1};
    cudaMalloc(&array_gpu, sizeof(cuda::std::array<uint64_t, 4>));
    cudaMemcpy(array_gpu, &array, sizeof(array), cudaMemcpyHostToDevice);
    cudaMemcpy(&array_cpu, array_gpu, sizeof(array), cudaMemcpyDeviceToHost);
    cudaFree(array_gpu);
    ASSERT_EQ(array, array_cpu);
}


TEST(CudaContainerSuite, CopyKernel)
{
    namespace knp_cuda = knp::backends::gpu::cuda;
    uint64_t *array_from = nullptr;
    uint64_t *array_to = nullptr;
    cudaMalloc(&array_from, 8 * sizeof(uint64_t));
    cudaMalloc(&array_to, 8 * sizeof(uint64_t));

    std::vector<uint64_t> vec_from = {3, 2, 4, 5, 1, 0, 4, 0};
    std::vector<uint64_t> vec_out(vec_from.size());
    cudaMemcpy(array_from, vec_from.data(), 8 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    knp_cuda::device_lib::copy_kernel<<<1, 8>>>(0, 8, array_to, array_from);
    cudaMemcpy(vec_out.data(), array_to, vec_from.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(array_from);
    cudaFree(array_to);
    ASSERT_EQ(vec_from, vec_out);
}


TEST(CudaContainerSuite, EqualKernel)
{
    // TODO: RAII !!!
    namespace knp_cuda = knp::backends::gpu::cuda;
    typedef uint64_t int_type;
    constexpr int num_values = 8;
    int_type *array = nullptr;
    int_type *array_same = nullptr;
    int_type *array_other = nullptr;
    cudaMalloc(&array, num_values * sizeof(int_type));
    cudaMalloc(&array_same, num_values * sizeof(int_type));
    cudaMalloc(&array_other, num_values * sizeof(int_type));

    std::vector<int_type> values = {1, 2, 1, 12, 9, 9, 3, 5};
    std::vector<int_type> other_values = {1, 2, 3, 4, 5, 6, 7, 8};

    uint64_t mem_size = num_values * sizeof(int_type);
    cudaMemcpy(array, values.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(array_same, values.data(), mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(array_other, other_values.data(), mem_size, cudaMemcpyHostToDevice);

    bool result = false;
    bool *gpu_result;

    cudaMalloc(&gpu_result, sizeof(bool));
    knp_cuda::device_lib::equal_kernel<<<1, 1>>>(array, array_same, num_values, gpu_result);
    cudaMemcpy(&result, gpu_result, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(result);
    knp_cuda::device_lib::equal_kernel<<<1, 1>>>(array, array_other, num_values, gpu_result);
    cudaMemcpy(&result, gpu_result, sizeof(bool), cudaMemcpyDeviceToHost);
    ASSERT_FALSE(result);
    cudaFree(array);
    cudaFree(array_same);
    cudaFree(array_other);
    cudaFree(gpu_result);
}


TEST(CudaContainerSuite, VectorPushBack)
{
    namespace knp_cuda = knp::backends::gpu::cuda;
    knp_cuda::device_lib::CudaVector<uint64_t> cuda_vec;

    ASSERT_EQ(cuda_vec.size(), 0);
    std::cout << cuda_vec << std::endl;
    cuda_vec.push_back(1);
    std::cout << cuda_vec << std::endl;
    cuda_vec.push_back(2);
    std::cout << cuda_vec << std::endl;
    cuda_vec.push_back(3);
    std::cout << cuda_vec << std::endl;

    ASSERT_EQ(cuda_vec.size(), 3);
    ASSERT_GE(cuda_vec.capacity(), 3);
    std::vector<uint64_t> exp_results{1, 2, 3};
    knp_cuda::device_lib::CudaVector res(exp_results.data(), exp_results.size());
    // ASSERT_EQ(cuda_vec, exp_results);
    ASSERT_EQ(cuda_vec[0], 1);
    ASSERT_EQ(cuda_vec[1], 2);
    ASSERT_EQ(cuda_vec[2], 3);
    ASSERT_EQ(cuda_vec, res);
}


TEST(CudaBaseSuite, CudaVectorConstruct)
{
    namespace knp_cuda = knp::backends::gpu::cuda;

    knp_cuda::device_lib::CudaVector<uint64_t> cuda_vec_1;
    knp_cuda::device_lib::CudaVector<uint64_t> cuda_vec_2(10);

    ASSERT_EQ(cuda_vec_1.size(), 0);
    ASSERT_EQ(cuda_vec_2.size(), 10);
}


TEST(CudaBackendSuite, CudaUidConversionTest)
{
    knp::core::UID orig_uid;
    knp::backends::gpu::cuda::UID cuda_uid = knp::backends::gpu::cuda::to_gpu_uid(orig_uid);
    knp::core::UID restored_uid = knp::backends::gpu::cuda::to_cpu_uid(cuda_uid);
    ASSERT_EQ(orig_uid, restored_uid);
}


TEST(CudaBackendSuite, MessagesTest)
{
    namespace knp_cuda = knp::backends::gpu::cuda;

    knp_cuda::SpikeMessage message_1;
    knp_cuda::SynapticImpactMessage message_2;
    ASSERT_EQ(message_1.neuron_indexes_.size(), 0);
    ASSERT_EQ(message_2.impacts_.size(), 0);
}


TEST(CudaBackendSuite, CudaHostSubscription)
{
    namespace knp_cuda = knp::backends::gpu::cuda;

    knp_cuda::UID receiver_uid = knp_cuda::to_gpu_uid(knp::core::UID{});
    knp_cuda::UID sender_1 = knp_cuda::to_gpu_uid(knp::core::UID{}), sender_2 = knp_cuda::to_gpu_uid(knp::core::UID{});
    ASSERT_NE(sender_1, sender_2);
    knp_cuda::Subscription<knp_cuda::SpikeMessage> subscription(receiver_uid, {sender_1});

    ASSERT_EQ(subscription.get_senders().size(), 1);
    ASSERT_TRUE(subscription.has_sender(sender_1));
    ASSERT_FALSE(subscription.has_sender(sender_2));
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
    // const auto &sub = ::cuda::std::get<knp::backends::gpu::cuda::Subscription<
    //                     knp::backends::gpu::cuda::SpikeMessage>>(sub_v);
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

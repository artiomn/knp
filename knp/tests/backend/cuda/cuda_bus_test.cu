/**
 * @file cuda_bus_test.cu
 * @brief CUDA backend test.
 * @kaspersky_support Vartenkov An.
 * @date 08.09.2025
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

#include <knp/core/message_bus.h>
#include <knp/core/population.h>
#include <knp/core/projection.h>

#include <generators.h>
#include <spdlog/spdlog.h>
#include <tests_common.h>

#include <functional>
#include <iostream>
#include <vector>


#include "../../../backends/gpu/cuda-backend/impl/cuda_lib/safe_call.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_lib/vector.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_bus/message_bus.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_bus/messaging.cuh"
#include "../../../backends/gpu/cuda-backend/impl/uid.cuh"


namespace knp::testing
{

struct MessageBusTandem
{
    MessageBusTandem() : cpu_(std::move(*knp::core::MessageBus::construct_bus())), gpu_(cpu_.create_endpoint())
    {}
    knp::core::MessageBus cpu_;
    knp::backends::gpu::cuda::CUDAMessageBus gpu_;
};


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
    call_and_check(cudaDeviceReset());
    knp_cuda::UID receiver_uid = knp_cuda::to_gpu_uid(knp::core::UID{});
    knp_cuda::UID sender_1 = knp_cuda::to_gpu_uid(knp::core::UID{}), sender_2 = knp_cuda::to_gpu_uid(knp::core::UID{});
    knp_cuda::UID sender_3 = knp_cuda::to_gpu_uid(knp::core::UID{}), sender_4 = knp_cuda::to_gpu_uid(knp::core::UID{});
    ASSERT_NE(sender_1, sender_2);
    knp_cuda::Subscription<knp_cuda::SpikeMessage> subscription(receiver_uid, {sender_1, sender_2, sender_3});
    ASSERT_EQ(subscription.get_senders().size(), 3);
    ASSERT_TRUE(subscription.has_sender(sender_2));
    ASSERT_FALSE(subscription.has_sender(sender_4));
}


TEST(CudaBackendSuite, BusSubscriptionsTest)
{
    namespace knp_cuda = knp::backends::gpu::cuda;
    MessageBusTandem message_buses;
    knp_cuda::UID uid_1{knp_cuda::new_uid()}, uid_2{knp_cuda::new_uid()};
    knp_cuda::UID uid_3{knp_cuda::new_uid()}, uid_4{knp_cuda::new_uid()};
    message_buses.gpu_.subscribe<knp_cuda::SpikeMessage>(uid_1, {uid_2, uid_3});
    ASSERT_EQ(message_buses.gpu_.get_subscriptions().size(), 1);
    message_buses.gpu_.subscribe<knp_cuda::SpikeMessage>(uid_2, {uid_3, uid_4});
    ASSERT_EQ(message_buses.gpu_.get_subscriptions().size(), 2);
    message_buses.gpu_.subscribe<knp_cuda::SpikeMessage>(uid_1, {uid_1});
    // TODO check corresponding subscription size, should become 3.
    ASSERT_EQ(message_buses.gpu_.get_subscriptions().size(), 2);
    // unsubscribing from a wrong type of messages, shouldn't change anything.
    message_buses.gpu_.unsubscribe<knp_cuda::SynapticImpactMessage>(uid_1);
    ASSERT_EQ(message_buses.gpu_.get_subscriptions().size(), 2);
    // now unsubscribing from the right type of messages.
    message_buses.gpu_.unsubscribe<knp_cuda::SpikeMessage>(uid_1);
    ASSERT_EQ(message_buses.gpu_.get_subscriptions().size(), 1);
}

} // namespace knp::testing

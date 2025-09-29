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


#include "../../../backends/gpu/cuda-backend/impl/cuda_lib/vector.cuh"
#include "../../../backends/gpu/cuda-backend/impl/cuda_lib/vector_kernels.cuh"
#include "../../../backends/gpu/cuda-backend/impl/uid.cuh"


// using Population = knp::backends::gpu::CUDABackend::PopulationVariants;
// using Projection = knp::backends::gpu::CUDABackend::ProjectionVariants;


namespace knp::testing
{

TEST(CudaVectorSuite, Memcpy)
{
    cudaDeviceReset();
    const uint64_t val = 112;
    uint64_t *val_gpu;
    uint64_t val_cpu = 0;
    cudaMalloc(&val_gpu, sizeof(uint64_t));
    cudaMemcpy(val_gpu, &val, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&val_cpu, val_gpu, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(val_gpu);
    ASSERT_EQ(val, val_cpu);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}


TEST(CudaVectorSuite, MemcpyArray)
{
    cudaDeviceReset();
    const cuda::std::array<uint64_t, 4> array{1, 2, 3, 4};
    cuda::std::array<uint64_t, 4> *array_gpu;
    cuda::std::array<uint64_t, 4> array_cpu{4, 3, 2, 1};
    cudaMalloc(&array_gpu, sizeof(cuda::std::array<uint64_t, 4>));
    cudaMemcpy(array_gpu, &array, sizeof(array), cudaMemcpyHostToDevice);
    cudaMemcpy(&array_cpu, array_gpu, sizeof(array), cudaMemcpyDeviceToHost);
    cudaFree(array_gpu);
    ASSERT_EQ(array, array_cpu);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}


TEST(CudaVectorSuite, CopyKernel)
{
    cudaDeviceReset();
    namespace knp_cuda = knp::backends::gpu::cuda;
    uint64_t *array_from = nullptr;
    uint64_t *array_to = nullptr;
    cudaMalloc(&array_from, 8 * sizeof(uint64_t));
    cudaMalloc(&array_to, 8 * sizeof(uint64_t));

    std::vector<uint64_t> vec_from = {3, 2, 4, 5, 1, 0, 4, 0};
    std::vector<uint64_t> vec_out(vec_from.size());
    cudaMemcpy(array_from, vec_from.data(), 8 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    knp_cuda::device_lib::copy_kernel<<<1, 8>>>(array_from, 0, 8, array_to);
    cudaMemcpy(vec_out.data(), array_to, vec_from.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(array_from);
    cudaFree(array_to);
    ASSERT_EQ(vec_from, vec_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

}


__global__ void copy_uid_kernel(size_t begin, size_t end, knp::backends::gpu::cuda::UID* data_to,
                                const knp::backends::gpu::cuda::UID* data_from)
{
    printf("Copy uid kernel, begin: %lu, end: %lu, sizeof data %lu\n", begin, end,
           sizeof(knp::backends::gpu::cuda::UID));
    if (end <= begin) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Copy kernel: index %lu, from %p to %p\n", i, data_from + begin + i, data_to + begin + i);
    if (i >= end - begin) return;
    new (data_to + begin + i) knp::backends::gpu::cuda::UID(*(data_from + begin + i));
    // *(data_to + begin + i) = *(data_from + begin + i);
}

TEST(CudaVectoSuite, CopyUidKernel)
{
    namespace knp_cuda = knp::backends::gpu::cuda;
    knp_cuda::UID *array_from = nullptr;
    knp_cuda::UID *array_to = nullptr;
    auto error = cudaGetLastError();
    if (error != cudaSuccess)
        std::cout << "ERROR BEFORE RESET: " << cudaGetErrorString(error) << std::endl;
    cudaDeviceReset();
    cudaMalloc(&array_from, 4 * sizeof(knp_cuda::UID));
    cudaMalloc(&array_to, 4 * sizeof(knp_cuda::UID));
    error = cudaGetLastError();
    if (error != cudaSuccess)
        std::cout << "ERROR 0: " << cudaGetErrorString(error) << std::endl;
    knp_cuda::UID uid1 = knp_cuda::to_gpu_uid(knp::core::UID{}), uid2 = knp_cuda::to_gpu_uid(knp::core::UID{});
    knp_cuda::UID uid3 = knp_cuda::to_gpu_uid(knp::core::UID{}), uid4 = knp_cuda::to_gpu_uid(knp::core::UID{});
    std::vector<knp_cuda::UID> vec_from{uid1, uid2, uid3, uid4};
    std::vector<knp_cuda::UID> vec_out(vec_from.size());
    error = cudaGetLastError();
    cudaMemcpy(array_from, vec_from.data(), 4 * sizeof(knp_cuda::UID), cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
        std::cout << "ERROR 1: " << cudaGetErrorString(error) << std::endl;

    // copy_uid_kernel<<<1, 4>>>(0, 4, array_to, array_from);
    knp_cuda::device_lib::copy_kernel<<<1, 4>>>(array_from, 0, 4, array_to);
    error = cudaGetLastError();
    if (error != cudaSuccess)
        std::cout << "ERROR 2: " << cudaGetErrorString(error) << std::endl;
    cudaMemcpy(vec_out.data(), array_to, vec_from.size() * sizeof(knp_cuda::UID), cudaMemcpyDeviceToHost);
    cudaFree(array_from);
    cudaFree(array_to);
    ASSERT_EQ(vec_from, vec_out);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}


TEST(CudaVectorSuite, EqualKernel)
{
    cudaDeviceReset();
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
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}


TEST(CudaVectorSuite, VectorPushBack)
{
    namespace knp_cuda = knp::backends::gpu::cuda;
    knp_cuda::device_lib::CUDAVector<uint64_t> cuda_vec;

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cuda_vec.size(), 0);
    std::cout << cuda_vec << std::endl;
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    cuda_vec.push_back(1);
    std::cout << cuda_vec << std::endl;
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    cuda_vec.push_back(2);
    std::cout << cuda_vec << std::endl;
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    cuda_vec.push_back(3);
    std::cout << cuda_vec << std::endl;
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cuda_vec.size(), 3);
    ASSERT_GE(cuda_vec.capacity(), 3);
    std::vector<uint64_t> exp_results{1, 2, 3};
    knp_cuda::device_lib::CUDAVector res(exp_results.data(), exp_results.size());
    // ASSERT_EQ(cuda_vec, exp_results);
    ASSERT_EQ(cuda_vec[0], 1);

    ASSERT_EQ(cuda_vec[1], 2);
    ASSERT_EQ(cuda_vec[2], 3);
    ASSERT_EQ(cuda_vec, res);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);

}


TEST(CudaVectorSuite, CUDAVectorConstruct)
{
    namespace knp_cuda = knp::backends::gpu::cuda;

    knp_cuda::device_lib::CUDAVector<uint64_t> cuda_vec_1;
    knp_cuda::device_lib::CUDAVector<uint64_t> cuda_vec_2(10);

    ASSERT_EQ(cuda_vec_1.size(), 0);
    ASSERT_EQ(cuda_vec_2.size(), 10);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}

}  // namespace knp::testing

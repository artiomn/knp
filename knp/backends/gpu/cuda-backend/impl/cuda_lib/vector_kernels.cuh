/**
 * @file vector_kernels.cuh
 * @brief CUDA STL-like vector implemented to work on GPU.
 * @kaspersky_support A. Vartenkov.
 * @date 06.07.2025
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

#include <cuda_runtime.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <utility>

#include "safe_call.cuh"
#include "cu_alloc.cuh"
#include "../uid.cuh"


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda::device_lib
{

// TODO : Move kernels to a different .h file.
template <class T, class Allocator>
__global__ void construct_kernel(T *data, size_t begin, size_t end)
{
    printf("Construct kernel, begin: %lu, end: %lu", begin, end);
    if (end <= begin) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end - begin) return;
    Allocator::construct(data + begin + i);
}

template __global__ void construct_kernel<UID, CuMallocAllocator<UID>>(UID *data, size_t begin, size_t end);


template<typename T, class Allocator>
__global__ void copy_construct_kernel(T* dest, const T src)
{
    Allocator::construct(dest, src);
}


template <class T>
__global__ void copy_kernel(const T* data_from, size_t begin, size_t end, T* data_to)
{
    printf("Copy kernel, begin: %lu, end: %lu, sizeof data %lu\n", begin, end, sizeof(T));
    if (end <= begin) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Copy kernel: index %lu, from %p to %p\n", i, data_from + begin + i, data_to + begin + i);
    if (i >= end - begin) return;
    new (data_to + begin + i) T(*(data_from + begin + i));
    // *(data_to + begin + i) = *(data_from + begin + i);
}


template __global__ void copy_kernel<UID>(const UID*, size_t, size_t, UID*);


template <class T>
__global__ void move_kernel(T* data_from, size_t begin, size_t end, T* data_to)
{
    if (end <= begin) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end - begin) return;
    *(data_to + begin + i) = ::cuda::std::move(*(data_from + begin + i));
}


template<class T, class Allocator>
__global__ void destruct_kernel(T* data, size_t begin, size_t end)
{
    if (end < begin) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end - begin) return;
    Allocator::destroy(data + begin + i);
}


template<typename T>
__global__ void equal_kernel(T *data_1, const T *data_2, size_t size, bool *equal)
{
    printf("Starting\n");
//        #ifdef __CUDA_ARCH__
    for (size_t i = 0; i < size; ++i)
    {
        printf("Index %lu\n", i);
        if (*(data_1 + i) != *(data_2 + i))
        {
            printf("Not equal!\n");
            *equal = false;
            return;
        }
    }
//#endif
    printf("Equal!\n");
    *equal = true;
};

} // namespace knp::backends::gpu::cuda::device_lib

/**
 * @file uid.cuh
 * @brief CUDA UID implementation.
 * @kaspersky_support Artiom N.
 * @date 28.03.2025
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

#include <cuda/std/array>
#include <cuda.h>
#include <knp/core/uid.h>


namespace knp::backends::gpu::cuda
{
constexpr int tag_size = 16;
using UID = ::cuda::std::array<std::uint8_t, tag_size>;

// bool __host__ is_equal(const UID &first, const UID &second)
// {
//     auto compare = [&first, &second] __global__ (bool &result)
//     {
//         result = (first == second);
//     };
//     bool result;
//     compare<<<1, 1>>>(result);
//     return result;
// }

inline cuda::UID uid_to_cuda(const knp::core::UID &source)
{
    cuda::UID result;

    for (size_t i = 0; i < source.tag.size(); ++i)
    {
        result[i] = *(source.tag.begin() + i);
    }

    return result;
}


cuda::UID to_gpu_uid(const knp::core::UID &uid)
{
    UID result;
    for (size_t i = 0; i < tag_size; ++i)
    {
        result[i] = uid.tag.data[i];
    }
    // cudaMemcpy(result.data(), uid.tag.data, sizeof(uint8_t) * tag_size, cudaMemcpyHostToHost);
    return result;
}


knp::core::UID to_cpu_uid(const cuda::UID &uid)
{
    knp::core::UID result;
    for (size_t i = 0; i < tag_size; ++i)
    {
        result.tag.data[i] = uid[i];
    }
    // cudaMemcpy(result.tag.data, uid.data(), sizeof(uint8_t) * tag_size, cudaMemcpyHostToHost);
    return result;
}

} // namespace knp::backends::gpu::cuda

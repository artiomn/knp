/**
 * @file gpu_cuda.cpp
 * @brief CUDA device class implementation.
 * @kaspersky_support Artiom N.
 * @date 24.02.2024
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

#include <knp/devices/gpu_cuda.h>

#include <spdlog/spdlog.h>

#include <exception>


namespace knp::devices::gpu
{

CUDA::CUDA(uint32_t gpu_num) : gpu_num_(gpu_num)
{
    if (const auto e_code = cudaGetDeviceProperties(&properties_, gpu_num_); e_code != cudaSuccess)
    {
        const auto err_msg = std::string("CUDA error during get device properties") + cudaGetErrorString(e_code);
        SPDLOG_ERROR("{} [{}]", err_msg.c_str(), e_code);
        throw std::runtime_error(err_msg);
    }

    static_assert(sizeof(boost::uuids::uuid) == sizeof(cudaUUID_t));
    std::copy_n(reinterpret_cast<const char*>(&properties_.uuid),
                Device::base_.uid_.tag.size(), Device::base_.uid_.tag.begin());
}


CUDA::CUDA(CUDA&& other)
    : gpu_num_{std::move(other.gpu_num_)},
      properties_{std::move(other.properties_)}
{
}


CUDA::~CUDA()
{
}


CUDA& CUDA::operator=(CUDA&& other) noexcept
{
    std::swap(gpu_num_, other.gpu_num_);
    std::swap(properties_, other.properties_);

    return *this;
}


knp::core::DeviceType CUDA::get_type() const
{
    return knp::core::DeviceType::GPU;
}


const std::string CUDA::get_name() const
{
    return properties_.name;
}


uint32_t CUDA::get_socket_number() const
{
    return gpu_num_;
}


float CUDA::get_power() const
{
    return 0;
}


KNP_DECLSPEC std::vector<CUDA> list_cuda_processors()
{
    int device_count = 0;

    if (auto e_code = cudaGetDeviceCount(&device_count); e_code != cudaSuccess)
    {
        const auto err_msg = std::string("CUDA error during get device properties") + cudaGetErrorString(e_code);
        SPDLOG_ERROR("{} [{}]", err_msg.c_str(), e_code);
        throw std::runtime_error(err_msg);
    }

    if (0 == device_count) return {};

    std::vector<CUDA> result;
    result.reserve(device_count);

    for (int i = 0; i < device_count; ++i) result.push_back(CUDA(i));

    return result;
}

}  // namespace knp::devices::gpu

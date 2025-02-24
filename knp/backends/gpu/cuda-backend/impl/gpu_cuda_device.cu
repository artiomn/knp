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

#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <spdlog/spdlog.h>

#include <exception>

#include <boost/uuid/name_generator.hpp>


namespace knp::devices::gpu
{

static constexpr const char* ns_uid = "0000-0000-0000-0001";


CUDA::CUDA(uint32_t gpu_num) : gpu_num_(gpu_num)
{
    gpu_name_ = "";  // pcm_instance->getCUDABrandString() + " " + pcm_instance->getCUDAFamilyModelString() + " " +
                     // std::to_string(cpu_num);
    Device::base_.uid_ = knp::core::UID(boost::uuids::name_generator(core::UID(ns_uid))(gpu_name_.c_str()));
}


CUDA::CUDA(CUDA&& other)
    : gpu_name_{std::move(other.gpu_name_)}
{
}


CUDA::~CUDA()
{
}


CUDA& CUDA::operator=(CUDA&& other) noexcept
{
    gpu_name_.swap(other.gpu_name_);
    return *this;
}


knp::core::DeviceType CUDA::get_type() const
{
    return knp::core::DeviceType::GPU;
}


const std::string& CUDA::get_name() const
{
    return gpu_name_;
}


uint32_t CUDA::get_socket_number() const
{
    return gpu_num_;
}


float CUDA::get_power() const
{
    return 0;
}


namespace cuda
{
KNP_DECLSPEC std::vector<CUDA> list_processors()
{
    std::vector<CUDA> result;
    result.reserve(1);

    return result;
}

}  //namespace cuda

}  // namespace knp::devices::gpu

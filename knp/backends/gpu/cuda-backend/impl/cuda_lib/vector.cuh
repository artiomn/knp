/**
 * @file vector.cuh
 * @brief CUDA STL-like vector implemented to work on GPU.
 * @kaspersky_support Artiom N.
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

#include <utility>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "safe_call.cuh"
#include "cu_alloc.cuh"


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda::device_lib
{

template <typename T, typename Allocator = CuMallocAllocator<T>>
class CudaVector
{
public:
    using value_type = T;

public:
    __device__ explicit CudaVector(size_t size = 0) : capacity_(size), size_(size), data_(nullptr)
    {
        data_ = allocator_.allocate(size_);
        for (size_t i = 0; i < size_; ++i) decltype(allocator_)::construct(data_ + i);
    }

    __device__ ~CudaVector()
    {
        for (size_t i = 0; i < size_; ++i) decltype(allocator_)::destroy(data_ + i);
        allocator_.deallocate(data_, size_);
    }

    // Copy constructor.
    __device__ CudaVector(const CudaVector& other) : capacity_(other.capacity_), size_(other.size_)
    {
        // Capacity.
        data_ = allocator_.allocate(capacity_ * sizeof(T));

        for (size_t i = 0; i < other.size_; ++i)
        {
            data_[i] = other.data_[i];
        }
    }

    // Move constructor.
    __device__ CudaVector(CudaVector&& other) noexcept :
        capacity_(other.capacity_), size_(other.size_), data_(other.data_)
    {
        capacity_ = other.capacity_;
        other.capacity_ = 0;

        size_ = other.size_;
        other.size_ = 0;

        data_ = other.data_;
        other.data_ = nullptr;
    }

    // Copy assignment operator.
    __device__ CudaVector& operator=(const CudaVector& other)
    {
        if (this != &other)
        {
            capacity_ = other.capacity_;
            size_ = other.size_;

            allocator_.deallocate(data_);
            // Capacity.
            data_ = allocator_.allocate(capacity_ * sizeof(T));

            for (size_t i = 0; i < other.size_; ++i)
            {
                data_[i] = other.data_[i];
            }
        }
        return *this;
    }

    // Move assignment operator.
    __device__ CudaVector& operator=(CudaVector&& other) noexcept
    {
        if (this != &other)
        {
            capacity_ = other.capacity_;
            other.capacity_ = 0;

            size_ = other.size_;
            other.size_ = 0;

            allocator_.deallocate(data_);
            data_ = other.data_;
            other.data_ = nullptr;
        }
        return *this;
    }

    // Sets size to 0 without reallocation or changing capacity.
    __device__ void clear() { size_ = 0; }

    __device__ T& operator[](size_t index)
    {
        return data_[index];
    }

    __device__ const T& operator[](size_t index) const
    {
        return data_[index];
    }

    __device__ size_t capacity() const
    {
        return capacity_;
    }

    __device__ size_t size() const
    {
        return size_;
    }

    __device__ T* data()
    {
        return data_;
    }

    __device__ void push_back(const T& value)
    {
        if (size_ == capacity_) reserve((size_ + 1) * 2);

        data_[size_++] = value;
    }

    __device__ void push_back(T&& value)
    {
        if (size_ == capacity_) reserve((size_ + 1) * 2);

        data_[size_++] = std::move(value);
    }

    __device__ T pop_back()
    {
        return data_[--size_];
    }

    __device__ void reserve(size_t new_capacity)
    {
        if (new_capacity <= capacity_) return;
        T* new_data = nullptr;

        // Capacity.
        new_data = allocator_.allocate(new_capacity * sizeof(T));

        for (size_t i = 0; i < size_; ++i)
        {
            new_data[i] = data_[i];
        }

        allocator_.deallocate(data_);
        data_ = new_data;
        capacity_ = new_capacity;
    }

    __device__ void resize(size_t new_size)
    {
        if (new_size < size_)
        {
            for (size_t i = new_size; i < size_ - new_size; ++i)
            {
                decltype(allocator_)::destroy(data_ + i);
            }

            reserve(new_size);
        }
        else if (new_size > size_)
        {
            reserve(new_size);

            for (size_t i = size_ - 1; i < new_size - size_; ++i)
            {
                decltype(allocator_)::construct(data_ + i);
            }
        }

        size_ = new_size;
    }

private:
    Allocator allocator_;
    T* data_;
    // Maximum elements count.
    size_t capacity_;
    // Current element.
    size_t size_;
};

} // namespace knp::backends::gpu::cuda::device_lib.

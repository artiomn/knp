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

#include <algorithm>
#include <iostream>
#include <vector>
#include <utility>

#include <cuda_runtime.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <spdlog/spdlog.h>

#include "get_blocks_config.cuh"
#include "safe_call.cuh"
#include "cu_alloc.cuh"

#include "vector_kernels.cuh"


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda::device_lib
{

template <typename T, typename Allocator = CuMallocAllocator<T>>
class CUDAVector
{
public:
    using value_type = T;
    using allocator_type = Allocator;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = typename allocator_type::pointer;
    using const_pointer = const typename allocator_type::pointer;
    using iterator = pointer;
    using const_iterator = const iterator;

public:
    __host__ __device__ explicit CUDAVector(size_type size = 0) : capacity_(size), size_(size), data_(nullptr)
    {
        #ifdef __CUDA_ARCH__
        data_ = allocator_.allocate(size_);
        for (size_type i = 0; i < size_; ++i) decltype(allocator_)::construct(data_ + i);
        #else
        if (size_)
        {
            auto [num_blocks, num_threads] = get_blocks_config(size_);
            data_ = allocator_.allocate(capacity_);
            construct_kernel<value_type><<<num_blocks, num_threads>>>(0, size_, data_, allocator_);
            cudaDeviceSynchronize();
        }
        #endif
    }

    __host__ __device__ CUDAVector(const value_type *vec, size_type size)
    {
        reserve(size);
        #ifdef __CUDA_ARCH__
        size_ = size;
        for (size_type i = 0; i < size_; ++i) allocator_.construct(data_ + i, *(vec + i));
        #else
        static_assert(std::is_trivially_copyable_v<value_type>);
        call_and_check(cudaMemcpy(data_, vec, size * sizeof(value_type), cudaMemcpyHostToDevice));
        size_ = size;
        #endif
    }

    __host__ CUDAVector(const std::vector<value_type> &vec)
    {
        static_assert(std::is_trivially_copyable_v<value_type>);
        clear();
        reserve(vec.size());
        call_and_check(cudaMemcpy(data_, vec.data(), vec.size() * sizeof(value_type), cudaMemcpyHostToDevice));
    }

    __host__ __device__ ~CUDAVector()
    {
        #ifdef __CUDA_ARCH__
        for (size_type i = 0; i < size_; ++i) decltype(allocator_)::destroy(data_ + i);
        allocator_.deallocate(data_, size_);
        #else
        auto [num_blocks, num_threads] = get_blocks_config(size_);
        destruct_kernel<<<num_blocks, num_threads>>>(0, size_, data_, allocator_);
        cudaDeviceSynchronize();
        if (capacity_) allocator_.deallocate(data_, capacity_);
        #endif
    }

    // Copy constructor.
    __host__ __device__ CUDAVector(const CUDAVector& other) : capacity_(other.capacity_), size_(other.size_)
    {
        // Capacity.
        #ifdef __CUDA_ARCH__
        data_ = allocator_.allocate(capacity_);

        for (size_type i = 0; i < other.size_; ++i)
        {
            data_[i] = other.data_[i];
        }
        #else
        data_ = allocator_.allocate(capacity_);
        auto [num_blocks, num_threads] = get_blocks_config(size_);
        copy_kernel<value_type><<<num_blocks, num_threads>>>(0, size_, data_, other.data_);
        cudaDeviceSynchronize();
        #endif
    }

    // Move constructor.
    __host__ __device__ CUDAVector(CUDAVector&& other) noexcept :
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
    __device__ CUDAVector& operator=(const CUDAVector& other)
    {
        if (this != &other)
        {
            for (size_type i = 0; i < size_; ++i) allocator_.destroy(data_ + i);
            allocator_.deallocate(data_);
            capacity_ = other.capacity_;
            size_ = other.size_;
            data_ = allocator_.allocate(capacity_);

            for (size_type i = 0; i < other.size_; ++i)
            {
                allocator_.construct(data_ + i);
                data_[i] = other.data_[i];
            }
        }
        return *this;
    }

    // Move assignment operator.
    __device__ CUDAVector& operator=(CUDAVector&& other) noexcept
    {
        if (this == &other) return *this;
        capacity_ = other.capacity_;
        other.capacity_ = 0;

        size_ = other.size_;
        other.size_ = 0;
        allocator_.deallocate(data_);
        data_ = other.data_;
        other.data_ = nullptr;

        return *this;
    }

    // template<class Other>
    // __host__ CUDAVector& operator=(const std::vector<Other> &vec)
    // {
    // }

    __host__ CUDAVector& operator=(const std::vector<value_type> &vec)
    {
        static_assert(std::is_trivially_copyable_v<value_type>);
        clear();
        reserve(vec.size());
        call_and_check(cudaMemcpy(data_, vec.data(), vec.size() * sizeof(value_type), cudaMemcpyHostToDevice));
        return *this;
    }

    __host__ __device__ bool operator==(const CUDAVector &other) const
    {
        if (size_ != other.size_) return false;

        #ifdef __CUDA_ARCH__
        for (size_type i = 0; i < size_; ++i) if (*(data_ + i) != *(other.data_ + i)) return false;
        return true;
        #else
        bool equal;
        bool *d_equal;
        cudaMalloc(&d_equal, sizeof(bool));
        equal_kernel<value_type><<<1, 1>>>(data_, other.data_, size_, d_equal);
        cudaMemcpy(&equal, d_equal, sizeof(bool), cudaMemcpyHostToDevice);
        cudaFree(d_equal);
        return equal;

        // return thrust::equal(thrust::device, data_, data_ + size_, other.data_);
        #endif
    }

    // Sets size to 0 without reallocation or changing capacity.
    __host__ __device__ void clear()
    {
        #ifdef __CUDA_ARCH__
        for (size_type i = 0; i < size_; ++i) allocator_.destroy(data_ + i);
        #else
        auto [num_blocks, num_threads] = get_blocks_config(size_);
        destruct_kernel<<<num_blocks, num_threads>>>(0, size_, data_, allocator_);
        cudaDeviceSynchronize();
        #endif
        size_ = 0;
    }

    __host__ __device__ bool set(uint64_t index, const value_type &value)
    {
        if (index >= size_) return false;
        #ifdef __CUDA_ARCH__
        data_[index] = value;
        #else
        static_assert(std::is_trivially_copyable_v<value_type>);
        cudaMemcpy(data_ + index, &value, sizeof(value_type), cudaMemcpyHostToDevice);
        #endif
        return true;
    }

    __host__ __device__ value_type operator[](size_type index) const
    {
        #ifdef __CUDA_ARCH__
        return data_[index];
        #else
        static_assert(std::is_trivially_copyable_v<value_type>);
        T result;
        call_and_check(cudaMemcpy(&result, data_ + index, sizeof(value_type), cudaMemcpyDeviceToHost));
        return result;
        #endif
    }

/*    __host__ __device__ T& operator[](size_t index)
    {
        #ifdef __CUDA_ARCH__
        return data_[index];
        #else
        // TODO: return thrust::reference (or smth. like this).
        static_assert(std::is_trivially_copyable_v<value_type>);
        T result;
        call_and_check(cudaMemcpy(&result, data_ + index, sizeof(value_type), cudaMemcpyDeviceToHost));
        return result;
        #endif
    }
*/
    __host__ __device__ size_type capacity() const
    {
        return capacity_;
    }

    __host__ __device__ size_type size() const
    {
        return size_;
    }

    __host__ __device__ bool empty() const
    {
        return 0 == size_;
    }

    __host__ __device__ value_type* data()
    {
        return data_;
    }

    __host__ __device__ void push_back(const value_type& value)
    {
        if (size_ == capacity_) reserve((size_ + 1) * 2);
        #ifdef __CUDA_ARCH__
        data_[size_++] = value;
        #else
        //static_assert(std::is_trivially_copyable_v<T>);
        call_and_check(cudaMemcpy(data_ + size_, &value, sizeof(value_type), cudaMemcpyHostToDevice));
        ++size_;

        // #ifdef DEBUG
        if constexpr (std::is_same_v<uint64_t, value_type>)
        {
            value_type val;
            cudaMemcpy(&val, data_ + size_ - 1, sizeof(value_type), cudaMemcpyDeviceToHost);
            std::cout << "Pushed back " << val << std::endl;
        }
        // #endif
        #endif
    }

    // __device__ void push_back(T&& value)
    // {
    //     if (size_ == capacity_) reserve((size_ + 1) * 2);

    //     data_[size_++] = std::move(value);
    // }

    __device__ value_type pop_back()
    {
        return data_[--size_];
    }

    __host__ __device__ void reserve(size_type new_capacity)
    {
        if (new_capacity <= capacity_) return;
        #ifdef __CUDA_ARCH__
        dev_reserve(new_capacity);
        #else
        T* new_data = allocator_.allocate(new_capacity);
        auto [num_blocks, num_threads] = get_blocks_config(size_);
        // Inefficient, better to use move.
        copy_kernel<value_type><<<num_blocks, num_threads>>>(0, size_, new_data, data_);
        cudaDeviceSynchronize();
        destruct_kernel<value_type><<<num_blocks, num_threads>>>(0, size_, data_, allocator_);
        cudaDeviceSynchronize();
        allocator_.deallocate(data_);
        data_ = new_data;
        capacity_ = new_capacity;
        #endif
    }

    __host__ __device__ void resize(size_type new_size)
    {
        if (new_size == size_) return;
        #ifdef __CUDA_ARCH__
        dev_resize(new_size);
        #else
        if (new_size > size_)
        {
            reserve(new_size);
            auto [num_blocks, num_threads] = get_blocks_config(new_size - size_);
            construct_kernel<value_type><<<num_blocks, num_threads>>>(data_, size_, new_size, allocator_);
        }
        else if (new_size < size_)
        {
            auto [num_blocks, num_threads] = get_blocks_config(size_ - new_size);
            destruct_kernel<value_type><<<num_blocks, num_threads>>>(data_, new_size, size_, allocator_);
        }
        cudaDeviceSynchronize();
        size_ = new_size;
        #endif
    }

    __host__ __device__ void erase(size_type index)
    {
        if (index >= size_) return;
        for (size_type i = index; i < size_ - 1; ++i)
            data_[i] = ::cuda::std::move(data_[i + 1]);
        --size_;
    }

    __host__ __device__ iterator begin() { return data_; }
    __host__ __device__ const iterator begin() const { return data_; }
    __host__ __device__ const_iterator cbegin() const { return data_; }

    __host__ __device__ iterator end() { return data_ + size_; }
    __host__ __device__ const iterator end() const { return data_ + size_; }
    __host__ __device__ const_iterator cend() const { return data_ + size_; }

private:
    __device__ void dev_reserve(size_type new_capacity)
    {
        if (new_capacity <= capacity_) return;
        T* new_data = nullptr;

        // Capacity.
        new_data = allocator_.allocate(new_capacity * sizeof(T));

        for (size_type i = 0; i < size_; ++i)
        {
            new_data[i] = data_[i];
            allocator_.destroy(data_ + i);
        }

        allocator_.deallocate(data_);

        data_ = new_data;
        capacity_ = new_capacity;
    }

    __device__ void dev_resize(size_type new_size)
    {
        if (new_size < size_)
        {
            for (size_type i = new_size; i < size_ - new_size; ++i)
            {
                decltype(allocator_)::destroy(data_ + i);
            }

            dev_reserve(new_size);
        }
        else if (new_size > size_)
        {
            dev_reserve(new_size);

            for (size_type i = size_ - 1; i < new_size - size_; ++i)
            {
                decltype(allocator_)::construct(data_ + i);
            }
        }

        size_ = new_size;
    }


    Allocator allocator_;
    pointer data_ = nullptr;
    // Maximum elements count.
    size_type capacity_ = 0;
    // Current element.
    size_type size_ = 0;
};


template<class T>
std::ostream &operator<<(std::ostream &stream, const CUDAVector<T> &vec)
{
    if (vec.size() == 0)
    {
        stream << "{}";
        return stream;
    }

    stream << "{";
    for (size_t i = 0; i < vec.size() - 1; ++i)
    {
        stream << vec[i] << ", ";
    }
    stream << vec[vec.size() - 1] << "}";
    return stream;
}
} // namespace knp::backends::gpu::cuda::device_lib

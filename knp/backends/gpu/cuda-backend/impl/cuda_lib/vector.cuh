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
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <spdlog/spdlog.h>

#include "safe_call.cuh"
#include "cu_alloc.cuh"


/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda::device_lib
{
constexpr size_t threads_per_block = 256;


// TODO : Move kernels to a different .h file.
template <class T, class Allocator>
__global__ void construct_kernel(size_t begin, size_t end, T* data, Allocator allocator)
{
    if (end <= begin) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end - begin) return;
    allocator.construct(data + begin + i);    
}


template<typename T, class Allocator>
__global__ void copy_construct_kernel(T* dest, const T src, Allocator allocator) 
{
    allocator.construct(dest, src);
}


template <class T>
__global__ void copy_kernel(size_t begin, size_t end, T* data_to, const T* data_from)
{
    if (end <= begin) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end - begin) return; 
    *(data_to + begin + i) = *(data_from + begin + i);
}


template <class T>
__global__ void move_kernel(size_t begin, size_t end, T* data_to, T* data_from)
{
    if (end <= begin) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end - begin) return;
    *(data_to + begin + i) = ::cuda::std::move(*(data_from + begin + i));

}


template<class T, class Allocator>
__global__ void destruct_kernel(size_t begin, size_t end, T* &data, Allocator allocator)
{
    if (end < begin) return;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end - begin) return;
    allocator.destroy(data + begin + i);
}


auto get_blocks_config(size_t num_total)
{
    size_t num_threads = std::min(num_total, threads_per_block);
    size_t num_blocks = (num_total + threads_per_block - 1) / threads_per_block;
    return std::make_pair(num_blocks, num_threads);
}

template <typename T, typename Allocator = CuMallocAllocator<T>>
class CudaVector
{
public:
    using value_type = T;
public:
    __host__ __device__ explicit CudaVector(size_t size = 0) : capacity_(size), size_(size), data_(nullptr)
    {
        #ifdef __CUDA_ARCH__
        data_ = allocator_.allocate(size_);
        for (size_t i = 0; i < size_; ++i) decltype(allocator_)::construct(data_ + i);  
        #else        
        if (size_) 
        {
            auto [num_blocks, num_threads] = get_blocks_config(size_);
            data_ = allocator_.allocate(capacity_);
            construct_kernel<<<num_blocks, num_threads>>>(0, size_, data_, allocator_);
            cudaDeviceSynchronize();
        }
        #endif
    }


    __host__ __device__ CudaVector(const T *vec, size_t size)
    {
        reserve(size);
        #ifdef __CUDA_ARCH__
        size_ = size;
        for (size_t i = 0; i < size_; ++i) allocator_.construct(data_ + i, *(vec + i));
        #else
        static_assert(std::is_trivially_copyable_v<T>);
        call_and_check(cudaMemcpy(data_, vec, size * sizeof(T), cudaMemcpyHostToDevice));
        size_ = size;
        #endif

    }
    

    __host__ __device__ ~CudaVector()
    {
        #ifdef __CUDA_ARCH__
        for (size_t i = 0; i < size_; ++i) decltype(allocator_)::destroy(data_ + i);
        allocator_.deallocate(data_, size_); 
        #else
        auto [num_blocks, num_threads] = get_blocks_config(size_);
        destruct_kernel<<<num_blocks, num_threads>>>(0, size_, data_, allocator_);
        cudaDeviceSynchronize();
        if (capacity_) allocator_.deallocate(data_, capacity_);
        #endif

    }

    // Copy constructor.
    __host__ __device__ CudaVector(const CudaVector& other) : capacity_(other.capacity_), size_(other.size_)
    {
        // Capacity.
        #ifdef __CUDA_ARCH__
        data_ = allocator_.allocate(capacity_);

        for (size_t i = 0; i < other.size_; ++i)
        {
            data_[i] = other.data_[i];
        }
        #else
        data_ = allocator_.allocate(capacity_);
        auto [num_blocks, num_threads] = get_blocks_config(size_);
        copy_kernel<<<num_blocks, num_threads>>>(0, size_, data_, other.data_);
        cudaDeviceSynchronize();
        #endif
    }

    // Move constructor.
    __host__ __device__ CudaVector(CudaVector&& other) noexcept :
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
            for (size_t i = 0; i < size_; ++i) allocator_.destroy(data_ + i);
            allocator_.deallocate(data_);
            capacity_ = other.capacity_;
            size_ = other.size_;
            data_ = allocator_.allocate(capacity_);

            for (size_t i = 0; i < other.size_; ++i)
            {
                allocator_.construct(data_ + i);
                data_[i] = other.data_[i]; 
            }
        }
        return *this;
    }

    // Move assignment operator.
    __device__ CudaVector& operator=(CudaVector&& other) noexcept
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
    // __host__ CudaVector& operator=(const std::vector<Other> &vec)
    // { 
    // }

    
    __host__ CudaVector& operator=(const std::vector<T> &vec)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        clear();
        reserve(vec.size());
        call_and_check(cudaMemcpy(data_, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice));
        return *this;
    }


    __host__ __device__ bool operator==(const CudaVector &other) const
    {
        if (size_ != other.size_) return false;
        
        #ifdef __CUDA_ARCH__
        for (size_t i = 0; i < size_; ++i) if (*(data_ + i) != *(other.data_ + i)) return false;
        return true;
        #else
        auto kernel = [] __global__ (T *data_1, const T *data_2, size_t size, bool &equal)
        {
            if (std::is_same_v<T, uint64_t>)
            {
                printf("%lu\n", *data_2);
                printf("%lu\n", *data_1);
            }
            for (size_t i = 0; i < size; ++i) if (*(data_1 + i) != *(data_2 + i)) 
            {
                equal = false;
                return;
            }
            equal = true;
        };
        bool equal;
        kernel<<<1, 1>>>(data_, other.data_, size_, equal);
        return equal;

        // return thrust::equal(thrust::device, data_, data_ + size_, other.data_);
        #endif
    }


    // Sets size to 0 without reallocation or changing capacity.
    __host__ __device__ void clear() 
    {
        #ifdef __CUDA_ARCH__
        for (size_t i = 0; i < size_; ++i) allocator_.destroy(data_ + i);
        #else
        auto [num_blocks, num_threads] = get_blocks_config(size_);
        destruct_kernel<<<num_blocks, num_threads>>>(0, size_, data_, allocator_);
        cudaDeviceSynchronize();
        #endif
        size_ = 0; 
    }

    __device__ T& operator[](size_t index)
    {
        return data_[index];
    }

    __host__ __device__ const T& operator[](size_t index) const
    {
        #ifdef __CUDA_ARCH__
        return data_[index];
        #else
        static_assert(std::is_trivially_copyable_v<T>());
        T result;
        call_and_check(cudaMemcpy(&result, data_ + index, sizeof(T), cudaMemcpyDeviceToHost));
        return result;
        #endif
    }

    __host__ __device__ size_t capacity() const
    {
        return capacity_;
    }

    __host__ __device__ size_t size() const
    {
        return size_;
    }

    __host__ __device__ T* data()
    {
        return data_;
    }

    __host__ __device__ void push_back(const T& value)
    {
        if (size_ == capacity_) reserve((size_ + 1) * 2);
        #ifdef __CUDA_ARCH__
        data_[size_++] = value;
        #else
        static_assert(std::is_trivially_copyable_v<T>);
        call_and_check(cudaMemcpy(data_ + size_, &value, sizeof(T), cudaMemcpyHostToDevice));
        ++size_;

        // #ifdef DEBUG
        if constexpr (std::is_same_v<uint64_t, T>) 
        {
            T val;
            cudaMemcpy(&val, data_ + size_ - 1, sizeof(T), cudaMemcpyDeviceToHost);
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

    __device__ T pop_back()
    {
        return data_[--size_];
    }

    __host__ __device__ void reserve(size_t new_capacity)
    {
        if (new_capacity <= capacity_) return;
        #ifdef __CUDA_ARCH__
        dev_reserve(new_capacity);
        #else
        T* new_data = allocator_.allocate(new_capacity);
        auto [num_blocks, num_threads] = get_blocks_config(size_);
        // Inefficient, better to use move.
        copy_kernel<<<num_blocks, num_threads>>>(0, size_, new_data, data_);
        cudaDeviceSynchronize();
        destruct_kernel<<<num_blocks, num_threads>>>(0, size_, data_, allocator_);
        cudaDeviceSynchronize();
        allocator_.deallocate(data_);
        data_ = new_data;
        capacity_ = new_capacity;
        #endif
    }

    __host__ __device__ void resize(size_t new_size) 
    {
        if (new_size == size_) return;
        #ifdef __CUDA_ARCH__
        dev_resize(new_size);
        #else
        if (new_size > size_)
        {
            reserve(new_size);
            auto [num_blocks, num_threads] = get_blocks_config(new_size - size_);
            construct_kernel<<<num_blocks, num_threads>>>(data_, size_, new_size, allocator_);
        }
        else if (new_size < size_)
        {
            auto [num_blocks, num_threads] = get_blocks_config(size_ - new_size);
            destruct_kernel<<<num_blocks, num_threads>>>(data_, new_size, size_, allocator_);
        }
        cudaDeviceSynchronize();
        size_ = new_size;
        #endif
    }

    __host__ __device__ void erase(size_t index)
    {
        if (index >= size_) return;
        for (size_t i = index; i < size_ - 1; ++i)
            data_[i] = ::cuda::std::move(data_[i + 1]);
        --size_;
    }

    __host__ __device__ T* begin() { return data_; }

    __host__ __device__ T* end() { return data_ + size_; }

private:
    __device__ void dev_reserve(size_t new_capacity)
    {
        if (new_capacity <= capacity_) return;
        T* new_data = nullptr;

        // Capacity.
        new_data = allocator_.allocate(new_capacity * sizeof(T));

        for (size_t i = 0; i < size_; ++i)
        {
            new_data[i] = data_[i];
            allocator_.destroy(data_ + i);
        }
        
        allocator_.deallocate(data_);
        
        data_ = new_data;
        capacity_ = new_capacity;
    }

    __device__ void dev_resize(size_t new_size)
    {
        if (new_size < size_)
        {
            for (size_t i = new_size; i < size_ - new_size; ++i)
            {
                decltype(allocator_)::destroy(data_ + i);
            }

            dev_reserve(new_size);
        }
        else if (new_size > size_)
        {
            dev_reserve(new_size);

            for (size_t i = size_ - 1; i < new_size - size_; ++i)
            {
                decltype(allocator_)::construct(data_ + i);
            }
        }

        size_ = new_size;
    }


    Allocator allocator_;
    T* data_ = nullptr;
    // Maximum elements count.
    size_t capacity_ = 0;
    // Current element.
    size_t size_ = 0;
};




} // namespace knp::backends::gpu::cuda::device_lib.

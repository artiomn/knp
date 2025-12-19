//
// Created by vartenkov on 18.12.25.
//

#pragma once

#include "kernels.cuh"


#define REGISTER_CUDA_VECTOR_NO_EXTRACT(data_type) \
    template __global__ void knp::backends::gpu::cuda::device_lib::construct_kernel<data_type, \
    knp::backends::gpu::cuda::device_lib::CuMallocAllocator<data_type>>(data_type *, size_t); \
    template __global__ void knp::backends::gpu::cuda::device_lib::copy_construct_kernel<data_type>( \
    data_type *, size_t, const data_type *); \
    template __global__ void knp::backends::gpu::cuda::device_lib::copy_kernel<data_type>(     \
    data_type *, size_t, const data_type *);                                         \
    template __global__ void knp::backends::gpu::cuda::device_lib::move_kernel<data_type>(       \
    data_type *, size_t, data_type*);        \
    template __global__ void knp::backends::gpu::cuda::device_lib::move_construct_kernel<data_type>( \
    data_type *, size_t, data_type *);                                          \
    template __global__ void knp::backends::gpu::cuda::device_lib::destruct_kernel<data_type,    \
    knp::backends::gpu::cuda::device_lib::CuMallocAllocator<data_type>>(data_type*, size_t)


// For any type used in CUDAVector you need to call this macro, or you'll get Error 0x62 (98).
#define REGISTER_CUDA_VECTOR_TYPE(data_type) \
    REGISTER_CUDA_VECTOR_NO_EXTRACT(data_type); \
    template __host__ data_type knp::backends::gpu::cuda::gpu_extract<data_type>(const data_type* );       \
    template __host__ void knp::backends::gpu::cuda::gpu_insert<data_type>(const data_type &, data_type *)

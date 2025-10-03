//
// Created by vartenkov on 23.09.25.
//
#pragma once
#include "safe_call.cuh"
#include "cu_alloc.cuh"

/**
 * @brief Namespace for CUDA implementations.
 */
namespace knp::backends::gpu::cuda
{
template<class T>
__host__ T gpu_extract(const T* gpu_val)
{
    T result;
    call_and_check(cudaMemcpy(&result, gpu_val, sizeof(T), cudaMemcpyDeviceToHost));
    if constexpr (!std::is_trivially_copyable<T>::value)
        result.actualize();
    cudaDeviceSynchronize(); // TODO not sure if needed;
    return result;
}


namespace detail
{
template<class T>
__global__ void actualize_kernel(T *val)
{
    val->actualize();
}
} // namespace detail


template<class T>
__host__ void gpu_insert(const T &cpu_val, T *gpu_target)
{
    call_and_check(cudaMemcpy(gpu_target, &cpu_val, sizeof(T), cudaMemcpyHostToDevice));
    if constexpr (!std::is_trivially_copyable<T>::value)
        detail::actualize_kernel<<<1, 1>>>(gpu_target);
    cudaDeviceSynchronize();
}

} // namespace knp::backends::gpu::cuda

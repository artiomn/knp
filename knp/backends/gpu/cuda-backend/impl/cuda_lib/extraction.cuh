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
__host__ T extract(const T* gpu_val)
{
    T result;
    call_and_check(cudaMemcpy(&result, gpu_val, sizeof(T), cudaMemcpyDeviceToHost));
    if constexpr (!std::is_trivially_copyable<T>::value)
        result.actualize();
    return result;
}

}

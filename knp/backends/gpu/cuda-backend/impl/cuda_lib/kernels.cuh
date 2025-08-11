#include "../uid.cuh"
#include "vector.cuh"

/**
 * @brief Namespace for CUDA message bus implementations.
 */
namespace knp::backends::gpu::cuda::device_lib
{

__global__ void has_sender_kernel(const UID &uid, device_lib::CudaVector<UID> senders, int *result);

} // namespace knp::backends::gpu::cuda::device_lib

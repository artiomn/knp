//
// Created by vartenkov on 26.09.25.
//


#include "messaging.cuh"
#include "subscription.cuh"
#include "../cuda_lib/extraction.cuh"

#include <type_traits>

REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SpikeMessage);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::SynapticImpactMessage);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::MessageVariant);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::Subscription);
REGISTER_CUDA_VECTOR_TYPE(knp::backends::gpu::cuda::UID);
REGISTER_CUDA_VECTOR_TYPE(uint64_t);

/**
 * @brief CUDA messaging namespace.
 */
namespace knp::backends::gpu::cuda
{
template<size_t index>
MessageVariant extract_message_by_index(const void *msg_ptr)
{
    return gpu_extract<boost::mp11::mp_at_c<AllCudaMessages, index>>(
            reinterpret_cast<const boost::mp11::mp_at_c<AllCudaMessages, index> *>(msg_ptr));
}


template<>
MessageVariant gpu_extract<MessageVariant>(const MessageVariant *message)
{
    int *type_gpu;
    const void **msg_gpu; // This is a gpu pointer to gpu pointer to gpu message.
    call_and_check(cudaMalloc(&type_gpu, sizeof(int)));
    call_and_check(cudaMalloc(&msg_gpu, sizeof(void *)));
    get_message_kernel<<<1, 1>>>(message, type_gpu, msg_gpu);
    int type;

    // This is a gpu pointer to gpu message. &msg_ptr is a cpu pointer to gpu pointer to gpu message.
    const void *msg_ptr;
    call_and_check(cudaMemcpy(&type, type_gpu, sizeof(int), cudaMemcpyDeviceToHost));
    call_and_check(cudaMemcpy(&msg_ptr, msg_gpu, sizeof(void *), cudaMemcpyDeviceToHost));
    call_and_check(cudaFree(type_gpu));
    call_and_check(cudaFree(msg_gpu));
    // Here we have a type index and a gpu pointer to message.
    MessageVariant result;
    // TODO: Remove crunchs.
    switch(type)
    {
        case 0: result = extract_message_by_index<0>(msg_ptr);
        case 1: result = extract_message_by_index<1>(msg_ptr);
    }
    return result;
}



namespace detail
{
    template <class Variant, class Instance>
    __global__ void make_variant_kernel(Variant *result, Instance *source)
    {
        new (result) Variant(*source);
    }
}


template<>
void gpu_insert<MessageVariant>(const MessageVariant &cpu_source, MessageVariant *gpu_target)
{
    ::cuda::std::visit([gpu_target](const auto &val)
    {
        using ValueType = std::decay_t<decltype(val)>;
        ValueType *buffer;
        call_and_check(cudaMalloc(&buffer, sizeof(ValueType)));
        gpu_insert(val, buffer);
        detail::make_variant_kernel<<<1, 1>>>(gpu_target, buffer);
        call_and_check(cudaFree(buffer));
    }, cpu_source);
}

} // namespace knp::backends::gpu::cuda

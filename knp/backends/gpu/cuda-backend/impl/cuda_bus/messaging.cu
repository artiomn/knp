//
// Created by vartenkov on 26.09.25.
//


#include "messaging.cuh"
#include "subscription.cuh"
#include "../cuda_lib/extraction.cuh"

#include <type_traits>


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



/**
 * @brief Extracts a subscription from GPU to CPU.
 * @param gpu_subscription a pointer to GPU subscription.
 * @return host-based subscription.
 */
template<class MessageT>
__host__ Subscription<MessageT> gpu_extract<Subscription<MessageT>>(const Subscription<MessageT> *gpu_subscription)
{
    Subscription<MessageT> result;
    cudaMemcpy(&result, gpu_subscription, sizeof(Subscription<MessageT>), cudaMemcpyDeviceToHost);
    result.actualize();
    return result;
}


// TODO: Code duplication, consider fixing it, maybe through template recursion. Like this?
// template<int I, class AllTypes, class TypesVariant = boost::mp11::mp_rename<AllSubscriptions, ::cuda::std::variant>>
//__host__ MessageVariant extract_variant_recursive(int type, const void* val_ptr) {
//    if constexpr (I == boost::mp11::mp_size<AllTypes>::value) {
//        return TypesVariant{}; // Base case
//    } else {
//        if (type == I) {
//            using ValueType = boost::mp11::mp_at_c<AllTypes, I>;
//            return extract<ValueType>(reinterpret_cast<const ValueType*>(msg_ptr));
//        }
//        return extract_variant_recursive<I + 1>(type, val_ptr);
//    }
//}


template<size_t index>
SubscriptionVariant extract_subscription_by_index(const void *sub_ptr)
{
    return gpu_extract<boost::mp11::mp_at_c<AllSubscriptions, index>>(
            reinterpret_cast<const boost::mp11::mp_at_c<AllSubscriptions, index> *>(sub_ptr));
}


template<>
SubscriptionVariant gpu_extract<SubscriptionVariant>(const SubscriptionVariant *sub_variant)
{
    SPDLOG_TRACE("Extracting subscription from GPU");
    int *type_gpu;
    const void **sub_gpu; // This is a gpu pointer to gpu pointer to gpu subscription.
    call_and_check(cudaMalloc(&type_gpu, sizeof(int)));
    call_and_check(cudaMalloc(&sub_gpu, sizeof(void *)));
    SPDLOG_TRACE("Calling get_subscription_kernel");
    get_subscription_kernel<<<1, 1>>>(sub_variant, type_gpu, sub_gpu);
    cudaDeviceSynchronize();
    SPDLOG_TRACE("Got subscription from gpu");
    int type;

    // This is a gpu pointer to gpu message. &msg_ptr is a cpu pointer to gpu pointer to gpu message.
    const void *sub_ptr;
    call_and_check(cudaMemcpy(&type, type_gpu, sizeof(int), cudaMemcpyDeviceToHost));
    call_and_check(cudaMemcpy(&sub_ptr, sub_gpu, sizeof(void *), cudaMemcpyDeviceToHost));
    call_and_check(cudaFree(type_gpu));
    call_and_check(cudaFree(sub_gpu));
    SPDLOG_TRACE("Copied pointed subscription to host.");
    // Here we have a type index and a gpu pointer to message.
    SubscriptionVariant result;
    // TODO: Remove crunchs.
    switch(type)
    {
        case 0: result = extract_subscription_by_index<0>(sub_ptr); break;
        case 1: result = extract_subscription_by_index<1>(sub_ptr); break;
        default: assert(false);
    }
    SPDLOG_TRACE("Successfully extracted subscription");
    return result;
}


template<>
void gpu_insert(const SubscriptionVariant &sub_var, SubscriptionVariant *gpu_target)
{
    ::cuda::std::visit([gpu_target](const auto &val)
                  {
                      using ValueType = std::decay_t<decltype(val)>;
                      ValueType *buffer;
                      call_and_check(cudaMalloc(&buffer, sizeof(ValueType)));
                      gpu_insert(val, buffer);
                      detail::make_variant_kernel<<<1, 1>>>(gpu_target, buffer);
                      call_and_check(cudaFree(buffer));
                  }, sub_var);
}


} // namespace knp::backends::gpu::cuda

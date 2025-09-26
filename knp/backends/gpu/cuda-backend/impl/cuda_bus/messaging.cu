//
// Created by vartenkov on 26.09.25.
//


#include "messaging.cuh"
#include "subscription.cuh"
#include "../cuda_lib/extraction.cuh"


/**
 * @brief CUDA messaging namespace.
 */
namespace knp::backends::gpu::cuda
{
template<size_t index>
MessageVariant extract_message_by_index(const void *msg_ptr)
{
    return extract<boost::mp11::mp_at_c<AllMessages, index>>(
            reinterpret_cast<const boost::mp11::mp_at_c<AllMessages, index> *>(msg_ptr));
}


template<>
MessageVariant extract<MessageVariant>(const MessageVariant *message)
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
    switch(type)
    {
        case 0: result = extract_message_by_index<0>(msg_ptr);
        case 1: result = extract_message_by_index<1>(msg_ptr);
    }
    return result;
}

/**
 * @brief Extracts a subscription from GPU to CPU.
 * @param gpu_subscription a pointer to GPU subscription.
 * @return host-based subscription.
 */
template<class MessageT>
__host__ Subscription<MessageT> extract<Subscription<MessageT>>(const Subscription<MessageT> *gpu_subscription)
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
    return extract<boost::mp11::mp_at_c<AllSubscriptions, index>>(
            reinterpret_cast<const boost::mp11::mp_at_c<AllSubscriptions, index> *>(sub_ptr));
}


template<>
SubscriptionVariant extract<SubscriptionVariant>(const SubscriptionVariant *sub_variant)
{
    int *type_gpu;
    const void **sub_gpu; // This is a gpu pointer to gpu pointer to gpu subscription.
    call_and_check(cudaMalloc(&type_gpu, sizeof(int)));
    call_and_check(cudaMalloc(&sub_gpu, sizeof(void *)));
    get_subscription_kernel<<<1, 1>>>(sub_variant, type_gpu, sub_gpu);
    int type;

    // This is a gpu pointer to gpu message. &msg_ptr is a cpu pointer to gpu pointer to gpu message.
    const void *sub_ptr;
    call_and_check(cudaMemcpy(&type, type_gpu, sizeof(int), cudaMemcpyDeviceToHost));
    call_and_check(cudaMemcpy(&sub_ptr, sub_gpu, sizeof(void *), cudaMemcpyDeviceToHost));
    call_and_check(cudaFree(type_gpu));
    call_and_check(cudaFree(sub_gpu));
    // Here we have a type index and a gpu pointer to message.
    SubscriptionVariant result;
    switch(type)
    {
        case 0: result = detail::extract_subscription_by_index<0>(sub_ptr);
        case 1: result = detail::extract_subscription_by_index<1>(sub_ptr);
    }
    return result;
}


} // namespace knp::backends::gpu::cuda

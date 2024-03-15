/**
 * @file message_header.cpp
 * @brief Message header Python bindings.
 * @author Artiom N.
 * @date 01.02.2024
 */
#include "common.h"


#if defined(_KNP_IN_CORE)

py::class_<core::messaging::MessageHeader>("MessageHeader", "Common header for messages.")
    .def_readwrite(
        "sender_uid", &core::messaging::MessageHeader::sender_uid_, "UID of the object that sent the message")
    .def_readwrite("send_time", &core::messaging::MessageHeader::send_time_, "Index of the network execution step")
    .def(py::self_ns::str(py::self));

#endif
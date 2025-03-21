/**
 * @file message_bus_zmq_impl.h
 * @brief Message bus ZeroMQ implementation header.
 * @kaspersky_support Artiom N.
 * @date 31.03.2023
 * @license Apache 2.0
 * @copyright © 2024 AO Kaspersky Lab
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

#include <message_bus_impl.h>
#include <spdlog/spdlog.h>

#include <string>

#include <zmq.hpp>


namespace knp::core::messaging::impl
{

/**
 * @brief Internal message bus class, not intended for user code.
 */
class MessageBusZMQImpl : public MessageBusImpl
{
public:
    MessageBusZMQImpl();

    /**
     * @brief Send a message from one socket to another.
     */
    size_t step() override;

    /**
     * @brief Create an endpoint that can be used for message exchange.
     * @return new endpoint.
     */
    [[nodiscard]] MessageEndpoint create_endpoint() override;

private:
    zmq::recv_result_t poll(zmq::message_t &message);
    bool isit_id(const zmq::recv_result_t &recv_result) const { return recv_result.value() == 5; }

private:
    /**
     * @brief Router socket address.
     */
    // cppcheck-suppress unusedStructMember
    std::string router_sock_address_;

    /**
     * @brief Publish socket address.
     */
    // cppcheck-suppress unusedStructMember
    std::string publish_sock_address_;

    /**
     * @brief Messaging context.
     */
    zmq::context_t context_;

    /**
     * @brief Router socket.
     */
    zmq::socket_t router_socket_;

    /**
     * @brief Publish socket.
     */
    zmq::socket_t publish_socket_;
};


}  // namespace knp::core::messaging::impl

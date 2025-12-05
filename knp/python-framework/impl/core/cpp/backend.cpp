/**
 * @file backend.cpp
 * @brief Python bindings for common Backend class.
 * @kaspersky_support Artiom N.
 * @date 01.02.2024
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

#include <filesystem>
#include <string>

#include "backend_wrapper.h"
#include "common.h"

//#if defined(KNP_IN_CORE)

// py::class_<std::vector<core::AllPopulationsVariant>>("AllPopulations")
//     .def(py::vector_indexing_suite<std::vector<core::AllPopulationsVariant>>() );

// py::class_<std::vector<core::AllProjectionsVariant>>("AllProjections")
//     .def(py::vector_indexing_suite<std::vector<core::AllProjectionsVariant>>() );


#define INSTANCE_PY_BACKEND_SUBSCRIBE_METHOD_IMPL(n, template_for_instance, message_type)                   \
    if (BOOST_PP_STRINGIZE(message_type) == class_obj_name)                                                 \
    {                                                                                                       \
        SPDLOG_TRACE("Backend subscribing to {}...", class_obj_name);                                       \
        self.subscribe<core::messaging::message_type>(receiver, py_iterable_to_vector<core::UID>(senders)); \
        return;                                                                                             \
    }

void export_backend()
{
    // "Abstract" class.
    py::class_<BackendWrapper, std::shared_ptr<knp::core::Backend>, boost::noncopyable>(
        "Backend", "The Backend class is the base class for backends.", py::no_init)
        .def(
            "load_all_populations",
            make_handler(
                [](core::Backend &self, const py::list &populations)
                {
                    using PT = core::AllPopulationsVariant;
                    self.load_all_populations(
                        std::vector<PT>(py::stl_input_iterator<PT>(populations), py::stl_input_iterator<PT>()));
                }),
            "Add populations to backend.")
        .def(
            "load_all_projections",
            make_handler(
                [](core::Backend &self, const py::list &projections)
                {
                    using PT = core::AllProjectionsVariant;
                    self.load_all_projections(
                        std::vector<PT>(py::stl_input_iterator<PT>(projections), py::stl_input_iterator<PT>()));
                }),
            "Add projections to backend.")
        .def(
            "load_all_projections", py::pure_virtual(&core::Backend::load_all_projections),
            "Add projections to backend.")
        .def(
            "remove_projections", py::pure_virtual(&core::Backend::remove_projections),
            "Remove projections with given UIDs from the backend.")
        .def(
            "remove_populations", py::pure_virtual(&core::Backend::remove_populations),
            "Remove populations with given UIDs from the backend.")
        // .def("get_current_devices",
        // py::pure_virtual(static_cast<std::vector<std::unique_ptr<core::Device>>&(core::Backend::*)()>(&core::Backend::get_current_devices)),
        //      "Get a list of devices on which the backend runs a network.")
        // .def("get_current_devices", py::pure_virtual(static_cast<const
        // std::vector<std::unique_ptr<core::Device>>&(core::Backend::*) const
        // ()>(&core::Backend::get_current_devices)),
        //      "Get a list of devices on which the backend runs a network.")
        .def("get_devices", py::pure_virtual(&core::Backend::get_devices))
        .def(
            "select_devices", py::pure_virtual(&core::Backend::select_devices),
            "Select devices on which to run the backend.")
        // .def( "get_message_endpoint",
        // py::pure_virtual(static_cast<core::MessageEndpoint&(core::Backend::*)()>(&core::Backend::get_message_endpoint)),
        // "Get message endpoint.") .def("get_message_endpoint", static_cast<const
        // core::MessageEndpoint&(core::Backend::*)() const>(& core::Backend::get_message_endpoint), "Get message
        // endpoint.")
        .def(
            "start", static_cast<void (core::Backend::*)()>(&core::Backend::start),
            "Start network execution on the backend.")
        .def(
            "start",
            make_handler(
                [](core::Backend &self, py::object &run_predicate)
                {
                    self.start(
                        [&run_predicate](core::Step step)
                        {
                            auto res = py::call<py::object>(run_predicate.ptr(), step);
                            return py::extract<bool>(res);
                        });
                }),
            "Start network execution on the backend.")
        .def(
            "start",
            make_handler(
                [](core::Backend &self, py::object &pre_step, py::object &post_step)
                {
                    self.start(
                        [&pre_step](core::Step step)
                        {
                            auto res = py::call<py::object>(pre_step.ptr(), step);
                            return py::extract<bool>(res);
                        },
                        [&post_step](core::Step step)
                        {
                            auto res = py::call<py::object>(post_step.ptr(), step);
                            return py::extract<bool>(res);
                        });
                }),
            "Start network execution on the backend.")
        .def("stop", &core::Backend::stop, "Stop network execution on the backend.")
        .def("get_step", &core::Backend::get_step, "Get current step.")
        .def("stop_learning", &core::Backend::stop_learning, "Stop learning.")
        .def("start_learning", &core::Backend::start_learning, "Restart learning.")
        .def(
            "subscribe",
            make_handler(
                [](core::Backend &self, const py::object &msg_class, const core::UID &receiver,
                   const py::list &senders)  // -> py::object
                {
                    const auto class_obj_name = get_py_class_name(msg_class);

                    SPDLOG_TRACE("Message class name: {}.", class_obj_name);

                    // cppcheck-suppress unknownMacro
                    BOOST_PP_SEQ_FOR_EACH(
                        INSTANCE_PY_BACKEND_SUBSCRIBE_METHOD_IMPL, "", BOOST_PP_VARIADIC_TO_SEQ(ALL_MESSAGES))

                    PyErr_SetString(PyExc_TypeError, "Passed object is not a message class.");
                    py::throw_error_already_set();

                    throw std::runtime_error("Incorrect class.");
                }),
            "Subscribe internal endpoint to messages.")
        .def("_init", &core::Backend::_init, "Initialize backend before starting network execution.")
        .def("_step", &core::Backend::_step, "Make one network execution step.")
        .def("_uninit", &core::Backend::_uninit, "Set backend to the uninitialized state.")
        .add_property(
            "message_bus",
            py::make_function(
                static_cast<core::MessageBus &(core::Backend::*)()>(&core::Backend::get_message_bus),  // NOLINT
                py::return_internal_reference<>()),
            "Get message bus used by backend.")
        .add_property(
            "message_endpoint",
            py::make_function(
                static_cast<core::MessageEndpoint &(core::Backend::*)()>(
                    &core::Backend::get_message_endpoint),  // NOLINT
                py::return_internal_reference<>()),
            "Get message endpoint.")
        .add_property("uid", make_handler([](core::Backend &self) { return self.get_uid(); }), "Get backend UID.")
        .add_property("running", &core::Backend::running, "Get network execution status.");

    //py::implicitly_convertible<knp::core::Backend, >();

    //register_direct_converter<knp::core::Backend>();
    py::register_ptr_to_python<std::shared_ptr<knp::core::Backend>>();
}
//#endif

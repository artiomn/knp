/**
 * @file base_framework.cpp
 * @brief Python bindings for C++ framework.
 * @kaspersky_support Artiom N.
 * @date 21.02.24
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

#include <knp/framework/backend_loader.h>

//#include "../../core/cpp/backend_wrapper.h"
#include "common.h"
#include "exports.h"


//// Anonymous namespace is necessary: without it DLL loading error under Windows is happened.
//static std::shared_ptr<Backend> load_backend(
//    cpp_framework::BackendLoader& loader, const py::object& backend_path)
//{
//     loader.load(py::extract<std::string>(backend_path)());
//}


template <typename T>
void register_direct_converter()
{
    // Need to register converter.
    // Without this extract from the different module can't convert Python object to C++ object.
    py::converter::registry::insert(
        [](PyObject* p) { return static_cast<void*>(p); }, py::type_id<T>(),
        &py::converter::registered_pytype_direct<T>::get_pytype);
}


BOOST_PYTHON_MODULE(KNP_FULL_LIBRARY_NAME)
{
    //PyImport_AppendInittab("_knp_python_framework_core", &PyInit__knp_python_framework_base_framework);
    py::import("knp.core._knp_python_framework_core");
    // auto path_type = py::import("pathlib.Path");

    py::implicitly_convertible<std::string, std::filesystem::path>();

    //py::implicitly_convertible<std::shared_ptr<knp::core::Backend>, BackendWrapper>();

    //register_direct_converter<knp::core::Backend>();
    py::register_ptr_to_python<std::shared_ptr<knp::core::Backend>>();

    py::class_<cpp_framework::BackendLoader>(
        "BackendLoader", "The BackendLoader class is a definition of a backend loader.")
        // // py::return_value_policy<py::manage_new_object>()
        //.def("load", &cpp_framework::BackendLoader::load, "Load backend")
        .def("load", &load_backend, "Load backend")
        .def("is_backend", &cpp_framework::BackendLoader::is_backend, "Check if the specified path points to a backend")
        .staticmethod("is_backend");

    export_input_channel();
    export_model();
    export_model_executor();
    export_model_loader();
    export_network();
    export_network_io();
    export_observers();
    export_output_channel();
}

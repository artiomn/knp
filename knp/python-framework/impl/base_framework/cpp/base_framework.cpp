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

#include "common.h"
#include "input_channel_utility.h"
#include "model_executor_utility.h"
#include "model_loader_utility.h"
#include "model_utility.h"
#include "network_utility.h"
#include "observer_utility.h"
#include "output_channel_utility.h"


// Anonymous namespace is necessary: without it DLL loading error under Windows is happened.
std::shared_ptr<knp::core::Backend> load_backend(cpp_framework::BackendLoader& loader, const py::object& backend_path)
{
    return loader.load(py::extract<std::string>(backend_path)());
}


BOOST_PYTHON_MODULE(KNP_FULL_LIBRARY_NAME)
{
#define KNP_IN_BASE_FW
    // auto path_type = py::import("pathlib.Path");

    py::implicitly_convertible<std::string, std::filesystem::path>();
    py::register_ptr_to_python<std::shared_ptr<knp::core::Backend>>();

    py::class_<cpp_framework::BackendLoader>(
        "BackendLoader", "The BackendLoader class is a definition of a backend loader.")
        // // py::return_value_policy<py::manage_new_object>()
        .def("load", &cpp_framework::BackendLoader::load, "Load backend")
        .def("load", &load_backend, "Load backend")
        .def("is_backend", &cpp_framework::BackendLoader::is_backend, "Check if the specified path points to a backend")
        .staticmethod("is_backend");

#include "input_channel.cpp"   // NOLINT
#include "model.cpp"           // NOLINT
#include "model_executor.cpp"  // NOLINT
#include "model_loader.cpp"    // NOLINT
#include "network.cpp"         // NOLINT
#include "network_io.cpp"      // NOLINT
#include "observer.cpp"        // NOLINT
#include "output_channel.cpp"  // NOLINT

#undef KNP_IN_BASE_FW
}

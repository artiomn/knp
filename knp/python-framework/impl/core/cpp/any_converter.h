/**
 * @file any_converter.h
 * @brief `std::any` for Python.
 * @kaspersky_support Artiom N.
 * @date 19.02.2024
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

// #include <object.h>
//
// #include <vector>
//
// #include "common.h"
//
//
//// Custom exceptions.
// struct BadAnyCastError : std::exception
//{
//     const char* what() const throw() { return "Bad \"any\" cast!"; }
// };
//
//// Set python exceptions.
//// void translate(const std::exception& e)
////{
////    // cppcheck-suppress knownPointerToBool
////    if (dynamic_cast<const BadAnyCastError*>(&e)) PyErr_SetString(PyExc_TypeError, e.what());
////}
//
//
// struct to_python_any
//{
//    static PyObject* convert(const std::any& obj) { return py::incref(py::object(std::any_cast<PyObject>(obj)).ptr());
//    }
//};
//
//
// struct from_python_any
//{
//    static void* convertible(PyObject* obj_ptr)
//    {
//        try
//        {
//            // return py::extract<std::any>(obj_ptr).has_value() ? obj_ptr : nullptr;
//            return nullptr;
//        }
//        // Without `try catch` it still raises a TypeError exception.
//        // But this enables you to customize your error message.
//        catch (...)
//        {
//            throw BadAnyCastError();
//        }
//    }
//
//    static void construct(PyObject* obj_ptr, boost::python::converter::rvalue_from_python_stage1_data* data)
//    {
//        const std::any value = py::extract<std::any>(obj_ptr);
//
//        if (value.has_value())
//        {
//            void* storage = ((py::converter::rvalue_from_python_storage<std::any>*)data)->storage.bytes;
//            new (storage) std::any(value);
//            data->convertible = storage;
//        }
//        else
//        {
//            data->convertible = nullptr;
//        }
//    }
//
//    from_python_any() { py::converter::registry::push_back(&convertible, &construct, py::type_id<std::any>()); }
//};

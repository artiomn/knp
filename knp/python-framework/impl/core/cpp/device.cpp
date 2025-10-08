/**
 * @file device.cpp
 * @brief Python bindings for device description.
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
#if defined(KNP_IN_CORE)

py::enum_<core::DeviceType>("DeviceType")
    .value("CPU", core::DeviceType::CPU)
    .value("GPU", core::DeviceType::GPU)
    .value("NPU", core::DeviceType::NPU);

//py::class_<core::Device>("Device", "The Device class is the base class for devices supported by the device library.");

#endif

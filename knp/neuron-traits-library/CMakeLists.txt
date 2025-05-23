#[[
© 2024 AO Kaspersky Lab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
]]

cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
    # Suppress Boost warning.
    cmake_policy(SET CMP0167 OLD)
endif()

project(knp-neuron-traits VERSION "${KNP_VERSION}" LANGUAGES C CXX
        DESCRIPTION "Kaspersky Neuromorphic Platform neuron traits library"
        HOMEPAGE_URL "https://neuro.kaspersky.ru/neyromorfnye-tekhnologii/")

include(knp-functions)

find_package(Boost ${KNP_BOOST_MIN_VERSION} REQUIRED)

set(${PROJECT_NAME}_PUBLIC_INCLUDE_DIR "knp/neuron-traits")

file(GLOB_RECURSE ${PROJECT_NAME}_headers include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}/*.h)

add_library("${PROJECT_NAME}"
        INTERFACE
        ${${PROJECT_NAME}_headers}
)

add_library(KNP::Neuron::Traits ALIAS "${PROJECT_NAME}")

target_include_directories("${PROJECT_NAME}" INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

target_link_libraries("${PROJECT_NAME}" INTERFACE Boost::headers KNP::Meta)

if (KNP_INSTALL)
    set(COMPONENT_NAME "cpp-framework")

    install(TARGETS "${PROJECT_NAME}"
            EXPORT "${PROJECT_NAME}"
            COMPONENT "${COMPONENT_NAME}-dev")

    install(DIRECTORY "include/${${PROJECT_NAME}_PUBLIC_INCLUDE_DIR}"
            COMPONENT "${COMPONENT_NAME}-dev"
            DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/knp"
            FILES_MATCHING PATTERN "*.h")
endif()

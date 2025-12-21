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

#
# CMake version reading functions. Artiom N.(cl)2022
#

include_guard(GLOBAL)


function(read_version_file filename)
    message(STATUS "Reading version file...")
    file(READ "${filename}" VERSION_FILE_RAW)

    string(STRIP "${VERSION_FILE_RAW}" KNP_VERSION)
    set(KNP_VERSION ${KNP_VERSION} PARENT_SCOPE)

    message(STATUS "Version from file: ${KNP_VERSION}.")
endfunction()


function(determine_version_with_git)
    message(STATUS "Determining version from Git in \"${CMAKE_CURRENT_SOURCE_DIR}\"...")
    execute_process(COMMAND git describe --tags
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            RESULT_VARIABLE GIT_EXECUTION_RESULT
            OUTPUT_VARIABLE GIT_PKG_VERSION_FULL
            ERROR_VARIABLE GIT_VERSION_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(GIT_EXECUTION_RESULT)
        message(STATUS "Could not retrieve version from command 'git describe --tags'.")
        return()
    endif()

    string(FIND ${GIT_PKG_VERSION_FULL} "." VALID_VERSION)
    if(VALID_VERSION EQUAL -1)
        message(WARNING "Version string ${GIT_PKG_VERSION_FULL} retrieved with `git describe` is invalid.")
        return()
    endif()
    set(KNP_VERSION ${GIT_PKG_VERSION_FULL} PARENT_SCOPE)
    message(STATUS "Version string determined with `git describe`: ${KNP_VERSION}.")
endfunction()


function(knp_get_version)
    determine_version_with_git()

    if (NOT KNP_VERSION)
        read_version_file("${CMAKE_CURRENT_SOURCE_DIR}/VERSION")
    endif()

    message(STATUS "Kaspersky Neuromorphic Platform version: ${KNP_VERSION}.")

    if (NOT KNP_VERSION)
        message(WARNING "Kaspersky Neuromorphic Platform version cannot be determined.")
        return()
    endif()

    # Parse the version information into pieces.
    string(REGEX REPLACE "v?([0-9]+\\.[0-9]+(\\.[0-9]+)?).*"  "\\1"
           KNP_VERSION "${KNP_VERSION}")

    set(KNP_VERSION "${KNP_VERSION}" PARENT_SCOPE)

    # set(KNP_VERSION ${KNP_VERSION} CACHE STRING "Kaspersky Neuromorphic Platform version" FORCE)
endfunction()

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
# CMake third-party support functions. Artiom N.(cl)2022
#

include_guard(GLOBAL)

# include(get_cpm)
include(CPM)


function(add_third_party module_name)
#    add_git_submodule("${CMAKE_CURRENT_SOURCE_DIR}/third-party/${module_name}")
    set(CPM_USE_LOCAL_PACKAGES ON)
    if(NOT KNP_ROOT_DIR)
        set(KNP_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    if(NOT "${module_name}" STREQUAL NAME)
        cpm_parse_add_package_single_arg("${module_name};${ARGN}" ARGN)
        # The shorthand syntax implies EXCLUDE_FROM_ALL and SYSTEM
        list(GET ARGN 1 _repo_name)
        get_filename_component(_repo_name ${_repo_name} NAME)
        CPMAddPackage(${ARGN}
                      EXCLUDE_FROM_ALL YES
                      SYSTEM YES)
                      # SOURCE_DIR "${KNP_ROOT_DIR}/third-party/${_repo_name}")
    else()
        list(GET ARGN 0 _m_name)
        CPMAddPackage("${module_name}" ${ARGN}
                      EXCLUDE_FROM_ALL YES
                      SYSTEM YES)
                      # SOURCE_DIR "${KNP_ROOT_DIR}/third-party/${_m_name}")
    endif()
endfunction()

cmake_minimum_required(VERSION 3.22)
project(MnistLearnExample)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Boost COMPONENTS program_options REQUIRED )

if (NOT TARGET KNP::BaseFramework::Core)
    find_package(knp-base-framework REQUIRED)
endif()

add_executable(MnistLearnExample main.cpp data_read.cpp construct_network.cpp evaluation.cpp)
target_link_libraries(MnistLearnExample PRIVATE KNP::BaseFramework::Core ${Boost_LIBRARIES})

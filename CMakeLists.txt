cmake_minimum_required(VERSION 3.10)
project(ct_segmentation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})
include_directories(${CMAKE_SOURCE_DIR})

add_executable(ct_segmentation main.cpp)
target_link_libraries(ct_segmentation ${ITK_LIBRARIES})
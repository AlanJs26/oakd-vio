cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)
project(VisualOdometry)

option(USE_CUDA "Using CUDA to accelerate" OFF)

if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif(USE_CUDA)

add_subdirectory(src)

add_executable( run src/main.cpp )
target_link_libraries( run ${OpenCV_LIBS} violib)

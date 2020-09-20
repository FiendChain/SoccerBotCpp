cmake_minimum_required(VERSION 3.6)
project(minimal)

set(EXECUTABLE_OUTPUT_PATH "build")
set(PROJECT_BINARY_DIR "bin")
set(PROJECT_SOURCE_DIR ".")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(vendor/)
include_directories(vendor/tensorflow/includes)
include_directories(src/)
add_executable(minimal src/minimal.cpp src/Model.cpp)
target_link_libraries(minimal
    ${CMAKE_SOURCE_DIR}/vendor/tensorflow/lib/libtensorflow-lite.a
    -lstdc++)
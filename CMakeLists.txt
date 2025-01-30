cmake_minimum_required(VERSION 3.23)
project(cuda_samples LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CUDA_ARCHITECTURES 86)

add_subdirectory(${PROJECT_SOURCE_DIR}/nvbench)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
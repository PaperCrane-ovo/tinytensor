cmake_minimum_required(VERSION 3.25)

project(tinytensor CUDA)

set(CMAKE_CUDA_ARCHITECTURES all)


add_executable(test test.cu)

target_include_directories(test
PRIVATE 
${PROJECT_SOURCE_DIR}/src
)



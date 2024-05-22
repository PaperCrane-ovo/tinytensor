cmake_minimum_required(VERSION 3.21)

project(tinytensor LANGUAGES CXX CUDA C)

set(CMAKE_CUDA_ARCHITECTURES all)

add_executable(tinytensor src/main.cu)


target_include_directories(tinytensor
PRIVATE 
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/third_party
)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(${PROJECT_NAME} CUDA::cublas CUDA::curand)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

include_directories(
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/third_party
)

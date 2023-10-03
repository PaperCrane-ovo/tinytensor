cmake_minimum_required(VERSION 3.21)

project(tinytensor LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES all)



add_executable(main
main.cpp
src/functions/ReLU.cu 
src/functions/Sigmoid.cu 
src/tensor/tensor.cpp
src/tensor/datastorage.cu
)


target_include_directories(main
PRIVATE 
${PROJECT_SOURCE_DIR}/src
E:/Program_Files/cuda_computing_toolkit_12.2/include
)



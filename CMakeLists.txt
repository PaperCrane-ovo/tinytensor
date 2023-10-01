cmake_minimum_required(VERSION 3.21)

project(tinytensor CUDA)

set(CMAKE_CUDA_ARCHITECTURES all)


add_executable(test 
test.cu 
src/functions/ReLU.cu 
src/functions/Sigmoid.cu 
src/tensor/tensor.cu
)


target_include_directories(test
PRIVATE 
${PROJECT_SOURCE_DIR}/src
)



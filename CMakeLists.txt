cmake_minimum_required(VERSION 3.25)

project(tinytensor CUDA)

set(CMAKE_CUDA_ARCHITECTURES all)


add_executable(test 
${PROJECT_SOURCE_DIR}/test.cu 
${PROJECT_SOURCE_DIR}/src/functions/ReLU.cu 
${PROJECT_SOURCE_DIR}/src/functions/Sigmoid.cu 
${PROJECT_SOURCE_DIR}/src/tensor/tensor.cu
)


target_include_directories(test
PRIVATE 
${PROJECT_SOURCE_DIR}/src
)



cmake_minimum_required(VERSION 3.21)

project(tinytensor LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES all)



add_executable(main
main.cu

)


target_include_directories(main
PRIVATE 
${PROJECT_SOURCE_DIR}/src
)



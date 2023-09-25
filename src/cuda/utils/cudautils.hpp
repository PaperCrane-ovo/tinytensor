#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>

// Use 512 or 256 threads per block
const int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}
// Define the grid stride looping
#define CUDA_KERNEL_LOOP(i, n)  for(int i = blockIdx.x * blockDim.x + threadIdx.x;i < (n);i += blockDim.x * gridDim.x)

#endif // CUDA_UTILS_H

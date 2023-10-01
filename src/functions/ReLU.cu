#include "ReLU.hpp"

using namespace Crane;
template<typename T>
ReLU<T>::ReLU(){}
template<typename T>
ReLU<T>::~ReLU(){}
template<typename T>
Tensor<T> ReLU<T>::forward(Tensor<T> input){
    Tensor<T> output(input.shape_, input.device_);
    if(input.device_ == Device::CPU){
        for(int i = 0; i < input.size_; i++){
            output.data_[i] = std::max((T)0, input.data_[i]);
        }
    }else{
        reluForwardKernel<<<CudaGetBlocks(input.size_),kCudaThreadsNum>>>(
            input.data_,output.data_,input.size_);
    }
}
template<typename T>
Tensor<T> ReLU<T>::backward(Tensor<T> grad){
    Tensor<T> output(grad.shape_, grad.device_);
    if(grad.device_ == Device::CPU){
        for(int i = 0; i < grad.size_; i++){
            output.data_[i] = std::max((T)0, grad.data_[i]);
        }
    }else{
        reluBackwardKernel<<<CudaGetBlocks(grad.size_),kCudaThreadsNum>>>(
            grad.data_,output.data_,grad.size_);
    }
    return output;
}

template<typename T>
std::string ReLU<T>::getName(){
    return "ReLU";
}

template <typename T>
__global__ void reluForwardKernel(T* input,T* output,int size){
    CUDA_KERNEL_LOOP(idx,size){
        output[idx] = max((T)0,input[idx]);
    }
}

template <typename T>
__global__ void reluBackwardKernel(T* grad,T* output,int size){
    CUDA_KERNEL_LOOP(idx,size){
        output[idx] = max((T)0,grad[idx]);
    }
}
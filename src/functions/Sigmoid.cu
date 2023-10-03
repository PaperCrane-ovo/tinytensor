#include "Sigmoid.hpp"

template <typename T>
Sigmoid<T>::Sigmoid(){}
template<typename T>
Sigmoid<T>::~Sigmoid(){}

template<typename T>
Tensor<T> Sigmoid<T>::forward(Tensor<T> input){
    Tensor<T> output(input.shape_,input.device_);
    this->output = Tensor<T>(input.shape_,input.device_);
    if(input.device_ == Device::CPU){
        for(int i = 0; i < input.size_; i++){
            output.data_[i] = 1.0/(1.0 + exp(-input.data_[i]));
            this->output.data_[i] = output.data_[i];
        }
    }else{
        sigmoidForwardKernel<<<CudaGetBlocks(input.size_),kCudaThreadsNum>>>(input.data_,output.data_,input.size_,this->output.data_);
    }
    return output;
}

template<typename T>
Tensor<T> Sigmoid<T>::backward(Tensor<T> input){
    Tensor<T> output(input.shape_,input.device_);
    if(input.device_ == Device::CPU){
        for(int i = 0; i < input.size_; i++){
            output.data_[i] = this->output.data_[i] * (1.0 - this->output.data_[i])*input.data_[i];
        }
    }else{
        sigmoidBackwardKernel<<<CudaGetBlocks(input.size_),kCudaThreadsNum>>>(input.data_,output.data_,input.size_,this->output.data_);
    }
    return output;
}

template<typename T>
std::string Sigmoid<T>::getName(){
    return "Sigmoid";
}

template<typename T>
__global__ void sigmoidForwardKernel(T* input, T* output, int size, T* thisOutput){
    CUDA_KERNEL_LOOP(idx, size){
        output[idx] = 1.0/(1.0 + exp(-input[idx]));
        thisOutput[idx] = output[idx];
    }
}

template<typename T>
__global__ void sigmoidBackwardKernel(T* input, T* output, int size, T* thisOutput){
    CUDA_KERNEL_LOOP(idx, size){
        output[idx] = thisOutput[idx] * (1.0 - thisOutput[idx])*input[idx];
    }
}

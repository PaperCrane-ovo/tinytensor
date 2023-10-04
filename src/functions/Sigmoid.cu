#include "Sigmoid.hpp"

template <typename T>
Sigmoid<T>::Sigmoid(){}
template<typename T>
Sigmoid<T>::~Sigmoid(){}

template<typename T>
Tensor<T> Sigmoid<T>::forward(Tensor<T> input){
    Tensor<T> output(input.getShape(),input.getDevice());
    this->output = Tensor<T>(input.getShape(),input.getDevice());
    if(input.getDevice() == Device::CPU){
        for(int i = 0; i < input.getSize(); i++){
            output[i] = 1.0/(1.0 + exp(-input[i]));
            this->output[i] = output[i];
        }
    }else{
        sigmoidForwardKernel<<<CudaGetBlocks(input.getSize()),kCudaThreadsNum>>>(input.data_->data,output.data_->data,input.getSize(),this->output.data_->data);
    }
    return output;
}

template<typename T>
Tensor<T> Sigmoid<T>::backward(Tensor<T> input){
    Tensor<T> output(input.getShape(),input.getDevice());
    if(input.getDevice() == Device::CPU){
        for(int i = 0; i < input.getSize(); i++){
            output[i] = this->output[i] * (1.0 - this->output[i])*input[i];
        }
    }else{
        sigmoidBackwardKernel<<<CudaGetBlocks(input.getSize()),kCudaThreadsNum>>>(input.data_->data,output.data_->data,input.getSize(),this->output.data_->data);
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

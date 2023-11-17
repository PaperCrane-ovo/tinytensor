#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "Module.cuh"
#include "../cuda/cudautils.cuh"
#include <iostream>
#include <cmath>



template<typename T>
class Sigmoid : public Module<T> {
    public:
        Sigmoid();
        ~Sigmoid();
        virtual Tensor<T> forward(const Tensor<T> &input);
        virtual Tensor<T> backward(const Tensor<T> &input);
        virtual std::string getName();
};

/***********************/

template <typename T>
Sigmoid<T>::Sigmoid(){}
template<typename T>
Sigmoid<T>::~Sigmoid(){}

template<typename T>
Tensor<T> Sigmoid<T>::forward(const Tensor<T>& input){
    mInput = input;
    mOutput = Tensor<T>(input.getShape(), input.getDevice());
    
    if(input.getDevice() == Device::CPU){
        for(int i = 0; i < input.getSize(); i++){
            mOutput[i] = 1.0/(1.0 + exp(-mInput[i]));
        }
    }else{
        sigmoidForwardKernel<<<CudaGetBlocks(input.getSize()),kCudaThreadsNum>>>(
            mInput.data_->data,
            mOutput.data_->data,
            input.getSize()
        );
    }
    return mOutput;
}

template<typename T>
Tensor<T> Sigmoid<T>::backward(const Tensor<T> &input){
    mOutGrad = input;
    mInGrad = Tensor<T>(input.getShape(), input.getDevice());
    
    if(input.getDevice() == Device::CPU){
        for(int i = 0; i < input.getSize(); i++){
            mInGrad[i] = mOutput[i] * (1.0 - mOutput[i]) * mOutGrad[i];
        }
    }else{
        sigmoidBackwardKernel<<<CudaGetBlocks(input.getSize()),kCudaThreadsNum>>>(
            mOutGrad.data_->data,
            mOutput.data_->data,
            mInGrad.data_->data,
            input.getSize());
    }
    return mInGrad;
}

template<typename T>
std::string Sigmoid<T>::getName(){
    return "Sigmoid";
}

template<typename T>
__global__ void sigmoidForwardKernel(T* input, T* output, int size){
    CUDA_KERNEL_LOOP(idx, size){
        output[idx] = 1.0/(1.0 + pow(E,-input[idx]));
    }
}

template<typename T>
__global__ void sigmoidBackwardKernel(T* outgrad, T* output,T* ingrad, int size){
    CUDA_KERNEL_LOOP(idx, size){
        ingrad[idx] = output[idx] * (1.0 - output[idx])*outgrad[idx];
    }
}

#endif /* SIGMOID_HPP */

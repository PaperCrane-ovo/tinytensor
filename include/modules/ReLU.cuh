#ifndef RELU_HPP
#define RELU_HPP

#include "../cuda/cudautils.cuh"
#include "Module.cuh"

#include <cmath>

template <typename T>

class ReLU:public Module<T>{
    public:
    ReLU();
    ~ReLU();
    virtual Tensor<T> forward(const Tensor<T> &input);
    virtual Tensor<T> backward(const Tensor<T> &input);
    virtual std::string getName();
    

};


/*******************************/


template<typename T>
ReLU<T>::ReLU(){}
template<typename T>
ReLU<T>::~ReLU(){}
template<typename T>
Tensor<T> ReLU<T>::forward(const Tensor<T> &input){
    mInput = input;
    mOutput = Tensor<T>(input.getShape(), input.getDevice());
    if(input.getDevice() == Device::CPU){
        for(int i = 0; i < input.getSize(); i++){
            mOutput[i] = std::max((T)0, mInput[i]);
        }
    }else{
        reluForwardKernel<<<CudaGetBlocks(input.getSize()),kCudaThreadsNum>>>(
            mInput.data_->data,mOutput.data_->data,mInput.getSize());
    }
    return mOutput;
}
template<typename T>
Tensor<T> ReLU<T>::backward(const Tensor<T> &grad){
    mOutGrad = grad;
    mInGrad = Tensor<T>(grad.getShape(), grad.getDevice());
    if(grad.getDevice() == Device::CPU){
        for(int i = 0; i < grad.getSize(); i++){
            mInGrad[i] = mOutput[i] > 0 ? mOutGrad[i] : 0;
        }
    }else{
        int size = grad.getSize();
        reluBackwardKernel<<<CudaGetBlocks(size),kCudaThreadsNum>>>(
            mOutGrad.data_->data,mOutput.data_->data,size,mInGrad.data_->data);
        
    }
    return mInGrad;
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
__global__ void reluBackwardKernel(T* input,T* outgrad,int size,T* ingrad){
    CUDA_KERNEL_LOOP(idx,size){
        ingrad[idx] = input[idx] > 0 ? outgrad[idx] : 0;
    }
}
#endif // RELU_HPP
#ifndef RELU_HPP
#define RELU_HPP

#include "../cuda/cudautils.cuh"
#include "ActivationFunction.cuh"

#include <cmath>

template <typename T>

class ReLU:public ActivationFunction<T>{
    public:
    ReLU();
    ~ReLU();
    virtual Tensor<T> forward(Tensor<T> &input);
    virtual Tensor<T> backward(Tensor<T> &input);
    virtual std::string getName();
    Tensor<T> output;

};


/*******************************/


template<typename T>
ReLU<T>::ReLU(){}
template<typename T>
ReLU<T>::~ReLU(){}
template<typename T>
Tensor<T> ReLU<T>::forward(Tensor<T> &input){
    Tensor<T> output(input.getShape(), input.getDevice());
    this->output = Tensor<T>(input.getShape(), input.getDevice());
    if(input.getDevice() == Device::CPU){
        for(int i = 0; i < input.getSize(); i++){
            output[i] = std::max((T)0, input[i]);
            this->output[i] = output[i];
        }
    }else{
        reluForwardKernel<<<CudaGetBlocks(input.getSize()),kCudaThreadsNum>>>(
            input.data_->data,output.data_->data,input.getSize(),this->output.data_->data);
    }
    return output;
}
template<typename T>
Tensor<T> ReLU<T>::backward(Tensor<T> &grad){
    Tensor<T> output(grad.getShape(), grad.getDevice());
    if(grad.getDevice() == Device::CPU){
        for(int i = 0; i < grad.getSize(); i++){
            output[i] = this->output[i] > 0 ? grad[i] : 0;
        }
    }else{
        reluBackwardKernel<<<CudaGetBlocks(grad.getSize()),kCudaThreadsNum>>>(
            grad.data_->data,output.data_->data,grad.getSize(),this->output.data_->data);
        
    }
    return output;
}

template<typename T>
std::string ReLU<T>::getName(){
    return "ReLU";
}

template <typename T>
__global__ void reluForwardKernel(T* input,T* output,int size,T* thisOutput){
    CUDA_KERNEL_LOOP(idx,size){
        output[idx] = max((T)0,input[idx]);
        thisOutput[idx] = output[idx];
    }
}

template <typename T>
__global__ void reluBackwardKernel(T* grad,T* output,int size,T* thisOutput){
    CUDA_KERNEL_LOOP(idx,size){
        output[idx] = thisOutput[idx] > 0 ? grad[idx] : 0;
    }
}
#endif // RELU_HPP
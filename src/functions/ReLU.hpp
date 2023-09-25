#ifndef RELU_HPP
#define RELU_HPP

#include "../cuda/utils/cudautils.hpp"
#include "ActivationFunction.hpp"

#include <cmath>
using namespace Crane;

template <typename T>

class ReLU:public ActivationFunction{
    public:
    ReLU();
    ~ReLU();
    virtual Tensor<T> forward(Tensor<T> input);
    virtual Tensor<T> backward(Tensor<T> input);
    virtual std::string getName();

    private:
    __global__ void reluForwardKernel(T* input, T* output, int size);
    __global__ void reluBackwardKernel(T* input, T* output, int size);


};

#endif // RELU_HPP
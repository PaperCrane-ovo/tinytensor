#ifndef RELU_HPP
#define RELU_HPP

#include "../cuda/cudautils.hpp"
#include "ActivationFunction.hpp"

#include <cmath>

template <typename T>

class ReLU:public ActivationFunction<T>{
    public:
    ReLU();
    ~ReLU();
    virtual Tensor<T> forward(Tensor<T> input);
    virtual Tensor<T> backward(Tensor<T> input);
    virtual std::string getName();

};
#endif // RELU_HPP
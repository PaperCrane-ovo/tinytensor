#ifndef ACTIVATIONFUNCTION_HPP_
#define ACTIVATIONFUNCTION_HPP_

#include <cmath>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include "../tensor/tensor.hpp"

template <typename T>
class ActivationFunction
{
public:
    virtual Tensor<T> forward(Tensor<T> input) = 0;
    virtual Tensor<T> backward(Tensor<T> input) = 0;
    virtual std::string getName() = 0;
};

#endif /* ACTIVATIONFUNCTION_HPP_ */
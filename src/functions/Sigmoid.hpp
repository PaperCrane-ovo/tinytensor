#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "ActivationFunction.hpp"
#include "../cuda/utils/cudautils.hpp"

#include <cmath>
using namespace Crane;
template<typename T>
class Sigmoid : public ActivationFunction<T> {
    public:
        Sigmoid();
        ~Sigmoid();
        virtual Tensor<T> forward(Tensor<T> input);
        virtual Tensor<T> backward(Tensor<T> input);
        virtual std::string getName();
    private:
        __global__ void sigmoidForwardKernel(T* input, T* output, int size,T* output2);
        __global__ void sigmoidBackwardKernel(T* input, T* output, int size,T* output2);
        Tensor<T> output;
};

#endif /* SIGMOID_HPP */

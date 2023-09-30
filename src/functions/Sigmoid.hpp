#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "ActivationFunction.hpp"
#include "../cuda/cudautils.hpp"

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
        Tensor<T> output;
};

#endif /* SIGMOID_HPP */

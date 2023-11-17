#ifndef Module_H
#define Module_H

#include "../tensor/tensor.cuh"

template <typename T>
class Module{
    public:
    virtual Tensor<T> forward(const Tensor<T> &input) = 0;
    virtual Tensor<T> backward(const Tensor<T> &outgrad) = 0;
    virtual std::string getName() = 0;
    Tensor<T> getWeightGrad() const {return mWeightGrad;}
    Tensor<T> getBiasGrad() const {return mBiasGrad;}

    protected:
    Tensor<T> mInput,mOutput,mInGrad,mOutGrad;
    Tensor<T> mWeight,mWeightGrad;
    Tensor<T> mBias,mBiasGrad;
    Device mDevice;
    

};

#endif
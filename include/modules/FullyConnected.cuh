#ifndef FullyConnected_H
#define FullyConnected_H

#include "../tensor/tensor.cuh"
#include "Module.cuh"

template<typename T>
class FullyConnected:public Module<T>{
public :
    virtual Tensor<T> forward(const Tensor<T>& input);
    virtual Tensor<T> backward(const Tensor<T>& grad);
    virtual std::string getName() {return "FullyConnected";}
    FullyConnected(int in_size,int out_size);
    FullyConnected(const Tensor<T> &w);
    FullyConnected(const Tensor<T> &w,const Tensor<T> &b);
    ~FullyConnected(){};
private:
    int mInFeature;
    int mOutFeature;
    bool mUseBias;
    int mBatchSize=0;
    
};

template<typename T>
FullyConnected<T>::FullyConnected(int in_size,int out_size){
    mInFeature=in_size;
    mOutFeature=out_size;
    mUseBias=true;
    mDevice=Device::CUDA;
    mWeight = Tensor<T>({mInFeature,mOutFeature},mDevice);
    // TODO: init weight

    if (mUseBias){
        mBias = Tensor<T>({mOutFeature},mDevice);
    }
}
template <typename T>
FullyConnected<T>::FullyConnected(const Tensor<T> &w){
    if (w.getShape().size()!=2){
        std::cout<<"Error: FullyConnected weight shape must be 2"<<std::endl;
    }
    mInFeature=w.shape()[0];
    mOutFeature=w.shape()[1];
    mUseBias=false;
    mDevice=w.getDevice();
    mWeight = w;
}
template <typename T>
FullyConnected<T>::FullyConnected(const Tensor<T> &w,const Tensor<T> &b){
    if (w.getShape().size()!=2){
        std::cout<<"Error: FullyConnected weight shape must be 2"<<std::endl;
    }
    mInFeature=w.getShape()[0];
    mOutFeature=w.getShape()[1];
    mUseBias=true;
    mDevice=w.getDevice();
    mWeight = w;
    mBias = b;
}


template<typename T>
Tensor<T> FullyConnected<T>::forward(const Tensor<T>& input){
    // input shape: [batch_size, in_feature]
    // output shape: [batch_size, out_feature]

    // TODO: check input shape

    // only support cuda now
    mInput = input;
    if (mBatchSize!=0 && mBatchSize!= mInput.getShape()[0] ){
        std::cerr<<"Error: batch size not match"<<std::endl;
        exit(1);
    } else if(mBatchSize==0){
        mBatchSize = mInput.getShape()[0];
    }
    mOutput = Tensor<T>({mBatchSize,mOutFeature},mDevice);
    Tensor<T>::matmul(1.0,mInput,mWeight,0.0,mOutput,false,false);
    if (mUseBias){
        Tensor<T> ones({mBatchSize,1},mDevice);
        ones.fill(1.0);
        Tensor<T>::matmul(1.0,ones,mBias,1.0,mOutput,false,false);
    }
    return mOutput;
}

template<typename T>
Tensor<T> FullyConnected<T>::backward(const Tensor<T>& grad){
    // grad shape: [batch_size, out_feature]
    // output shape: [batch_size, in_feature]
    mOutGrad = grad;
    mInGrad = Tensor<T>({mBatchSize,mInFeature},mDevice);
    Tensor<T>::matmul(1.0,mOutGrad,mWeight,0.0,mInGrad,false,true); // [batch_size, out_feature] * [out_feature, in_feature] = [batch_size, in_feature]
    mWeightGrad = Tensor<T>({mInFeature,mOutFeature},mDevice);
    Tensor<T>::matmul(1.0,mInput,mOutGrad,0.0,mWeightGrad,true,false); // [batch_size,in_feature]^T * [batch_size, out_feature] = [in_feature, out_feature]
    if (mUseBias){
        mBiasGrad = Tensor<T>({1,mOutFeature},mDevice);
        Tensor<T> ones({mBatchSize,1},mDevice);
        ones.fill(1.0);
        Tensor<T>::matmul(1.0,ones,mOutGrad,0.0,mBiasGrad,true,false);
    }
    return mInGrad;

}
#endif // FC_H
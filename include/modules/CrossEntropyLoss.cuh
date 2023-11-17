#ifndef CROSSENTROPYLOSS_CUH
#define CROSSENTROPYLOSS_CUH

#include "../tensor/tensor.cuh"
#include "Module.cuh"

template <typename T>
class CrossEntropyLoss : public Module<T>{
    public:
    virtual Tensor<T> forward(const Tensor<T>& input);
    virtual Tensor<T> backward(const Tensor<T>& grad);
    virtual std::string getName(){
        return "CrossEntropyLoss";}

    CrossEntropyLoss(int batchSize,int classNum):mBatchSize(batchSize),mClassNum(classNum){}
    CrossEntropyLoss(const Tensor<T>& target):target(target){}
    CrossEntropyLoss(){}

    private:
    Tensor<T> target;
    int mBatchSize,mClassNum;
    T mLoss;

};

template <typename T>
Tensor<T> CrossEntropyLoss<T>::forward(const Tensor<T>& input){
    auto shape = input.getShape();
    mBatchSize = shape[0];
    mClassNum = shape[1];
    mInput = input;
    mOutput = Tensor<T>({mBatchSize,1},input.getDevice());
    crossentropyloss_forward_kernel<<<CudaGetBlocks(mBatchSize),kCudaThreadsNum>>>(input.data_->data,mOutput.data_->data,target.data_->data,mBatchSize,mClassNum);

    mLoss = 0;
    mOutput.to(Device::CPU);
    for(int i=0;i<mBatchSize;i++){
        mLoss += mOutput[i];
    }
    mLoss /= (T)mBatchSize;
    mOutput.to(input.getDevice());
    return mOutput;
}

template <typename T>
Tensor<T> CrossEntropyLoss<T>::backward(const Tensor<T>& grad){
    mInGrad = Tensor<T>({mBatchSize,mClassNum},grad.getDevice());
    crossentropyloss_backward_kernel<<<CudaGetBlocks(mBatchSize*mClassNum),kCudaThreadsNum>>>(mInput.data_->data,mInGrad.data_->data,target.data_->data,mBatchSize,mClassNum);
    return mInGrad;
}


template <typename T>
__global__ void crossentropyloss_forward_kernel(T* input,T* output,T* target,int batchsize,int classnum)
{
    CUDA_KERNEL_LOOP(i,batchsize){
        int classid = target[i];
        if(classid<classnum && classid>=0){
            output[i] = -log(input[i*classnum+classid]);
        }else{
            output[i] = (T)0;
        }
    }
}

template <typename T>
__global__ void crossentropyloss_backward_kernel(T* input,T* inGrad,T* target,int batchsize,int classnum)
{
    CUDA_KERNEL_LOOP(i,batchsize*classnum){
        int batchid = i/classnum;
        int classid = i%classnum;
        int temp;
        if(classid == target[batchid]){
            temp = 1;
        }else{
            temp = 0;
        }
        inGrad[i] = input[i] - (T)temp;

    }
}

#endif
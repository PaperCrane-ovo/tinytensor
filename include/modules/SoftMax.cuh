#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH

#include "../tensor/tensor.cuh"
#include "Module.cuh"

#define E 2.718281828

template<typename T>
class SoftMax:public Module<T>{
    public:
        virtual Tensor<T> forward(const Tensor<T>& input);
        virtual Tensor<T> backward(const Tensor<T>& grad);
        virtual std::string getName(){return "SoftMax";}

    private:
        Tensor<T> mSum,mMax;
        int mBatchSize,mClassNum;
};

template<typename T>
Tensor<T> SoftMax<T>::forward(const Tensor<T>& input){
    auto shape = input.getShape();
    mBatchSize = shape[0];
    mClassNum = shape[1];
    mOutput = Tensor<T>(shape,input.getDevice());
    mSum = Tensor<T>({mBatchSize,1},input.getDevice());
    mMax = Tensor<T>({mBatchSize,1},input.getDevice());

    softmax_max_kernel<<<CudaGetBlocks(mBatchSize),kCudaThreadsNum>>>(input.data_->data,mMax.data_->data,mBatchSize,mClassNum);
    // 等待线程同步
    cudaDeviceSynchronize();
    softmax_exp_kernel<<<CudaGetBlocks(mBatchSize*mClassNum),kCudaThreadsNum>>>(input.data_->data,mMax.data_->data,mOutput.data_->data,mBatchSize,mClassNum);
    cudaDeviceSynchronize();
    softmax_sum_kernel<<<CudaGetBlocks(mBatchSize),kCudaThreadsNum>>>(mOutput.data_->data,mSum.data_->data,mBatchSize,mClassNum);
    cudaDeviceSynchronize();
    softmax_norm_kernel<<<CudaGetBlocks(mBatchSize*mClassNum),kCudaThreadsNum>>>(mSum.data_->data,mOutput.data_->data,mBatchSize,mClassNum);

    return mOutput;

}

template<typename T>
Tensor<T> SoftMax<T>::backward(const Tensor<T>& grad){
    Tensor<T> temp;
    return temp;
}

template<typename T>
__global__ void softmax_max_kernel(T*input,T* mMax,int n,int c){
    CUDA_KERNEL_LOOP(i,n){
        T max = input[i*c];
        for(int j=1;j<c;j++){
            if(input[i*c+j]>max){
                max = input[i*c+j];
            }
        }
        mMax[i] = max;
    }
}

template<typename T>
__global__ void softmax_exp_kernel(T* input,T* mMax,T* output,int n,int c){
    // substract max
    CUDA_KERNEL_LOOP(i,n*c){
        int row = i/c;
        output[i] = (T)pow(E,input[i]-mMax[row]);
    }
}

template <typename T>
__global__ void softmax_sum_kernel(T* input,T* sum,int n,int c){
    CUDA_KERNEL_LOOP(i,n){
        T sum_ = 0;
        for(int j=0;j<c;j++){
            sum_ += input[i*c+j];
        }
        sum[i] = sum_;
    }
}

template<typename T>
__global__ void softmax_norm_kernel(T* sum,T* output,int n,int c){
    CUDA_KERNEL_LOOP(i,n*c){
        int row = i/c;
        output[i] = output[i]/sum[row];
    }
}

#endif
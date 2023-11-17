#ifndef MAXPOOLING_CUH
#define MAXPOOLING_CUH

#include "../tensor/tensor.cuh"
#include "Module.cuh"

template<typename T>
class MaxPooling : public Module<T>{
    // kernel size = 2, stride = 2
    public:
        virtual Tensor<T> forward(const Tensor<T>& input);
        virtual Tensor<T> backward(const Tensor<T>& grad);
        virtual std::string getName(){
            return "MaxPooling";
        }

    private:
        int mBatchSize;
        int mChannel;
        int mInHeight;
        int mInWidth;
        int mOutHeight;
        int mOutWidth;
        Tensor<T> mMask;
        
};

template<typename T>
Tensor<T> MaxPooling<T>::forward(const Tensor<T>& input){
    // TODO: shape check
    auto shape = input.getShape();
    mBatchSize = shape[0];
    mChannel = shape[1];
    mInHeight = shape[2];
    mInWidth = shape[3];
    mOutHeight = mInHeight / 2 + (mInHeight % 2);
    mOutWidth = mInWidth / 2 + (mInWidth % 2);

    mInput = input;
    mOutput = Tensor<T>({mBatchSize,mChannel,mOutHeight,mOutWidth},input.getDevice());
    mMask = Tensor<T>({mBatchSize,mChannel,mInHeight,mInWidth},input.getDevice());

    int nthreads = mBatchSize * mChannel * mOutHeight * mOutWidth;
    maxpooling_forward_kernel<<<CudaGetBlocks(nthreads),kCudaThreadsNum>>>(input.data_->data,nthreads,mBatchSize,mChannel,mOutHeight,mOutWidth,mInHeight,mInWidth,mOutput.data_->data,mMask.data_->data);

    return mOutput;

}

template<typename T>
Tensor<T> MaxPooling<T>::backward(const Tensor<T>& grad){
    mOutGrad = grad;
    mInGrad = Tensor<T>({mBatchSize,mChannel,mInHeight,mInWidth},grad.getDevice());

    int nthreads = mBatchSize * mChannel * mOutHeight * mOutWidth;
    maxpooling_backward_kernel<<<CudaGetBlocks(nthreads),kCudaThreadsNum>>>(grad.data_->data,nthreads,mBatchSize,mChannel,mOutHeight,mOutWidth,mInHeight,mInWidth,mInGrad.data_->data,mMask.data_->data);

    return mInGrad;
}

template<typename T>
__global__ void maxpooling_forward_kernel(T*input,int nthreads,int n,int c,int h,int w,int H,int W,T*output,T* mask){
    int hw = h * w,chw = c * hw;
    CUDA_KERNEL_LOOP(i,nthreads){
        int nidx = i / chw;
        int cidx = (i % chw) / hw;
        int hidx = ((i % chw) % hw) / w;
        int widx = ((i % chw) % hw) % w;

        int new_cidx = nidx * c * H * W + cidx * H * W;
        T max;
        int max_idx;

        for(int j = 0;j<2;j++){
            for (int k = 0; k<2;k++){
                int new_hidx = hidx * 2 + j;
                int new_widx = widx * 2 + k;
                if(new_hidx < H && new_widx < W){
                    int new_idx = new_cidx + new_hidx * W + new_widx;
                    if(j == 0 && k == 0){
                        max = input[new_idx];
                        max_idx = new_idx;
                    }else{
                        if(input[new_idx] > max){
                            max = input[new_idx];
                            max_idx = new_idx;
                        }
                    }
                    mask[new_idx] = (T)0;
                }
            }
        }
        output[nidx * c * h * w + cidx * h * w + hidx * w + widx] = max;
        mask[max_idx] = (T)1;
    }
}

template<typename T>
__global__ void maxpooling_backward_kernel(T* outGrad,int nthreads,int n,int c,int h,int w,int H,int W,T*inGrad,T* mask){
    int hw = h * w,chw = c * hw;
    CUDA_KERNEL_LOOP(i,nthreads){
        int nidx = i / chw;
        int cidx = (i % chw) / hw;
        int hidx = ((i % chw) % hw) / w;
        int widx = ((i % chw) % hw) % w;

        T max = outGrad[nidx*chw + cidx*hw + hidx*w + widx];

        for(int j=0;j<2;j++){
            for(int k=0;k<2;k++){
                int new_hidx = hidx * 2 + j;
                int new_widx = widx * 2 + k;
                if(new_hidx < H && new_widx < W){
                    inGrad[nidx*c*H*W + cidx*H*W + new_hidx*W + new_widx] = mask[nidx*c*H*W + cidx*H*W + new_hidx*W + new_widx] * max;
                }
            }
        }
    }
}

#endif // MAXPOOLING_CUH

#ifndef CONV_CUH
#define CONV_CUH

#include "../tensor/tensor.cuh"
#include "Module.cuh"

template <typename T>
class Convolution : public Module<T>
{
public:
    virtual Tensor<T> forward(const Tensor<T> &input);
    virtual Tensor<T> backward(const Tensor<T> &grad);
    virtual std::string getName();
    Convolution(int in_channel, int out_channel);
    Convolution(const Tensor<T> &weight);
    Convolution(const Tensor<T> &weight, const Tensor<T> &bias);
    ~Convolution(){};

private:
    int mInChannel;
    int mOutChannel;
    int mKernelSize;
    int mStride;
    int mPadding;
    Tensor<T> mIm2Col;
    bool mUseBias = false;
    int mWidth, mHeight, mBatchSize;
    Tensor<T> mIm2ColGrad;
};

template <typename T>
std::string Convolution<T>::getName()
{
    return "Convolution";
}

template <typename T>
Convolution<T>::Convolution(int in_channel, int out_channel)
{
    mInChannel = in_channel;
    mOutChannel = out_channel;
    mKernelSize = 3;
    mStride = 1;
    mPadding = 0;
    mWeight = Tensor<T>({out_channel, in_channel, mKernelSize, mKernelSize}, Device::CUDA); // TODO: 初始化权重
    mWeight.randomfill();

    mUseBias = false;
}

template <typename T>
Convolution<T>::Convolution(const Tensor<T> &weight)
{
    mWeight = weight;
    mOutChannel = weight.getShape()[0];
    mInChannel = weight.getShape()[1];
    mKernelSize = weight.getShape()[2];
    mStride = 1;
    mPadding = 0;
    mUseBias = false;
}
template <typename T>
Convolution<T>::Convolution(const Tensor<T> &weight, const Tensor<T> &bias)
{
    mWeight = weight;
    mBias = bias;
    mOutChannel = weight.getShape()[0];
    mInChannel = weight.getShape()[1];
    mKernelSize = weight.getShape()[2];
    mStride = 1;
    mPadding = 0;
    mUseBias = true;
}

template <typename T>
Tensor<T> Convolution<T>::forward(const Tensor<T> &input)
{
    // input: [batch,in_channel,height,width]
    // output: [batch,out_channel,height,width]
    auto in_shape = input.getShape();
    mOutput = Tensor<T>({in_shape[0], mOutChannel, in_shape[2], in_shape[3]}, input.getDevice());
    mInput = input;
    mWidth = in_shape[3];
    mHeight = in_shape[2];
    mBatchSize = in_shape[0];
    mIm2Col = Tensor<T>({mBatchSize, mHeight * mWidth, mKernelSize * mKernelSize * mInChannel}, input.getDevice());
    mWeight.reShape({mOutChannel, mInChannel * mKernelSize * mKernelSize});
    // 首先需要做im2col
    // 每个batch在cpu上做im2col
    for (int i = 0; i < mBatchSize; i++)
    {
        Tensor<T> im = Tensor<T>::subtensor(input, {mInChannel, mHeight, mWidth}, i * mInChannel * mHeight * mWidth);
        Tensor<T> col = Tensor<T>::subtensor(mIm2Col, {mHeight * mWidth, mKernelSize * mKernelSize * mInChannel}, i * mHeight * mWidth);
        // 一共开mHeight*mWidth*mInChannel个线程
        im2col_kernel<<<CudaGetBlocks(mInChannel * mHeight * mWidth), kCudaThreadsNum>>>(im.data_->data, col.data_->data, mInChannel, mHeight, mWidth);
        Tensor<T> out = Tensor<T>::subtensor(mOutput, {mOutChannel, mHeight * mWidth}, i * mOutChannel * mHeight * mWidth);
        Tensor<T>::matmul(1.0, mWeight, col, 0.0, out, false, true);
        if (mUseBias)
        {
            Tensor<T> ones = Tensor<T>({1, mHeight * mWidth}, input.getDevice());
            ones.fill(1.0);
            Tensor<T>::matmul(1.0, mBias, ones, 1.0, out, true, false);
        }
    }
    return mOutput;
}

template <typename T>
Tensor<T> Convolution<T>::backward(const Tensor<T> &grad)
{
    mOutGrad = grad;
    mWeightGrad = Tensor<T>({mOutChannel, mInChannel * mKernelSize * mKernelSize}, grad.getDevice());
    mIm2ColGrad = Tensor<T>({mBatchSize, mHeight * mWidth, mKernelSize * mKernelSize * mInChannel}, grad.getDevice());
    mInGrad = Tensor<T>({mBatchSize, mInChannel, mHeight, mWidth}, grad.getDevice());
    mBiasGrad = Tensor<T>({mOutChannel, 1}, grad.getDevice());
    mWeightGrad.fill(0.0);

    for (int i = 0; i < mBatchSize; i++)
    {
        auto col = Tensor<T>::subtensor(mIm2Col, {mHeight * mWidth, mKernelSize * mKernelSize * mInChannel}, i * mHeight * mWidth * mInChannel * mKernelSize * mKernelSize);
        auto out = Tensor<T>::subtensor(mOutGrad, {mOutChannel, mHeight * mWidth}, i * mOutChannel * mHeight * mWidth);
        Tensor<T>::matmul(1.0, out, col, 1.0, mWeightGrad, false, false);
    }
    mWeightGrad.reShape({mOutChannel, mInChannel, mKernelSize, mKernelSize});
    mWeight.reShape({mOutChannel, mInChannel * mKernelSize * mKernelSize});

    for (int i = 0; i < mBatchSize; i++)
    {
        auto out = Tensor<T>::subtensor(mOutGrad, {mOutChannel, mHeight * mWidth}, i * mOutChannel * mHeight * mWidth);
        auto col = Tensor<T>::subtensor(mIm2ColGrad, {mHeight * mWidth, mKernelSize * mKernelSize * mInChannel}, i * mHeight * mWidth * mInChannel * mKernelSize * mKernelSize);
        Tensor<T>::matmul(1.0, out, mWeight, 0.0, col, true, false);
    }
    mInGrad.fill(0.0);

    for (int i = 0; i < mBatchSize; i++)
    {
        auto im = Tensor<T>::subtensor(mInGrad, {mInChannel, mHeight, mWidth}, i * mInChannel * mHeight * mWidth);
        auto col = Tensor<T>::subtensor(mIm2ColGrad, {mHeight * mWidth, mKernelSize * mKernelSize * mInChannel}, i * mHeight * mWidth * mInChannel * mKernelSize * mKernelSize);

        col2im_kernel<<<CudaGetBlocks(mInChannel * mHeight * mWidth), kCudaThreadsNum>>>(col.data_->data, im.data_->data, mInChannel, mHeight, mWidth);
    }

    if (mUseBias)
    {
        mBiasGrad.fill(0.0);
        for (int i = 0; i < mBatchSize; i++)
        {
            auto out = Tensor<T>::subtensor(mOutGrad, {mOutChannel, mHeight * mWidth}, i * mOutChannel * mHeight * mWidth);
            auto ones = Tensor<T>({1, mHeight * mWidth}, out.getDevice());
            ones.fill(1.0);
            Tensor<T>::matmul(1.0, ones, out, 1.0, mBiasGrad, false, true);
        }
    }
    return mInGrad;
}

template <typename T>
__global__ void im2col_kernel(T *im, T *col, int c, int h, int w)
{
    // 默认kernel_size为3，stride为1，padding为0
    int n = c * h * w, hw = h * w;
    CUDA_KERNEL_LOOP(i, n)
    {
        int channel_idx = i / hw;      // 通道id
        int height_idx = (i % hw) / w; // 高度id
        int width_idx = (i % hw) % w;  // 宽度id

        T *im_ptr = im + channel_idx * hw + i % hw;
        T *col_ptr = col + c * 3 * 3 * (i % hw) + channel_idx * 3 * 3;
        int cnt = 0;
        for (int j = -1; j <= 1; j++)
        {
            for (int k = -1; k <= 1; k++)
            {
                int h_id = height_idx + j;
                int w_id = width_idx + k;
                if (h_id >= 0 && h_id < h && w_id >= 0 && w_id < w)
                {
                    col_ptr[cnt] = im_ptr[j * w + k];
                }
                else
                {
                    col_ptr[cnt] = 0;
                }
                cnt++;
            }
        }
    }
}

template <typename T>
__global__ void col2im_kernel(T *col, T *im, int c, int h, int w)
{
    int n = c * h * w, hw = h * w;
    CUDA_KERNEL_LOOP(i, n)
    {
        int channel_idx = i / hw;      // 通道id
        int height_idx = (i % hw) / w; // 高度id
        int width_idx = (i % hw) % w;  // 宽度id

        T *im_ptr = im + channel_idx * hw + i % hw;
        T *col_ptr = col + c * 3 * 3 * (i % hw) + channel_idx * 3 * 3;
        int cnt = 0;
        for (int j = -1; j <= 1; j++)
        {
            for (int k = -1; k <= 1; k++)
            {
                int h_id = height_idx + j;
                int w_id = width_idx + k;
                if (h_id >= 0 && h_id < h && w_id >= 0 && w_id < w)
                {
                    im_ptr[j * w + k] = col_ptr[cnt];
                }
                cnt++;
            }
        }
    }
}

#endif
#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include "../cuda/cudautils.cuh"
#include "datastorage.cuh"
#include <array>
#include <cstring>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <algorithm>
#include <cublas_v2.h>
#include <curand.h>


template <typename T, typename = std::void_t<>>
class GetShape
{
public:
  static std::vector<int> get_shape() { return {}; }
};

template <typename T>
class GetShape<T, std::void_t<std::enable_if_t<std::is_array_v<T>>>>
{
public:
  static std::vector<int> get_shape()
  {
    // std::cout << typeid(T).name() << std::endl;
    auto size = GetShape<std::remove_extent_t<T>>::get_shape();
    size.push_back(std::extent_v<T>);
    return size;
  }
};

template <typename T>
class Tensor
{
public:
    Tensor();
    Tensor(std::vector<int> shape, std::string device);
    Tensor(std::vector<int> shape,Device device);
    Tensor(std::vector<int> shape,Device device,std::vector<T> data);
    Tensor(const Tensor &tensor);
    template <typename A, typename = std::enable_if_t<std::is_array_v<A>>>
    Tensor(const A &a);

    ~Tensor();
    // 获取张量的形状
    std::vector<int> getShape() const;
    // 获取张量的数据
    T &operator()(std::vector<int> index);
    T &operator[](int index);

    void Set(std::vector<int> index, T value);
    void print(std::string str);

    Tensor to(std::string device);
    Tensor to(Device device);
    Tensor cpu();
    Tensor gpu();

    int getSize() const;
    Device getDevice() const;
    std::shared_ptr<DataStorage<T>> data_;

    static void matmul(T alpha, const Tensor<T> &A, const Tensor<T> &B, T beta, Tensor<T> &C,bool transA,bool transB);
    void reShape(std::vector<int> shape);

    void fill(T value);
    void transpose();
    void randomfill();

    // TODO
    static Tensor<T> subtensor(const Tensor<T> &tensor,std::vector<int> shape,int offset);


private:
    std::vector<int> shape_;

    

    int Index(std::vector<int> indices) const;
};



/*****************************************/



template<typename T>
Tensor<T>::Tensor(){}

template<typename T>
Tensor<T>::Tensor(std::vector<int> shape,std::string device){
    shape_ = shape;
    int size = 1;
    for (int i = 0; i < shape.size(); i++){
        size *= shape[i];
    }
    Device device_ = device == "cuda" ? Device::CUDA : Device::CPU;
    data_ = std::make_shared<DataStorage<T>>(size,device_);
}

template<typename T>
Tensor<T>::Tensor(std::vector<int> shape,Device device,std::vector<T> data){
    shape_ = shape;
    int size = 1;
    for (int i = 0; i < shape.size(); i++){
        size *= shape[i];
    }
    data_ = std::make_shared<DataStorage<T>>(size,device);
    if(device == Device::CPU)
        std::memmove(data_->data, data.data(), size * sizeof(T));
    else
        cudaMemcpy(data_->data, data.data(), size * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
Tensor<T>::Tensor(std::vector<int> shape,Device device){
    shape_ = shape;
    int size = 1;
    for (int i = 0; i < shape.size(); i++){
        size *= shape[i];
    }
    data_ = std::make_shared<DataStorage<T>>(size,device);
}

//TODO: implement copy constructor
template<typename T>
Tensor<T>::Tensor(const Tensor<T>& tensor){
    shape_ = tensor.shape_;
    data_ = tensor.data_;
}

template <typename T>
template <typename A, typename>
Tensor<T>::Tensor(const A &a) : shape_(GetShape<A>::get_shape())
{
    int s= 1;
    for (int i = 0; i < shape_.size(); i++){
        s *= shape_[i];
    }
    data_ = std::make_shared<DataStorage<T>>(s,Device::CUDA);
    cudaMemcpy(data_->data, a, s * sizeof(T), cudaMemcpyHostToDevice);
    // 将size逆序
    std::reverse(shape_.begin(), shape_.end());
}

template <typename T>
Tensor<T>::~Tensor(){}



template <typename T>
std::vector<int> Tensor<T>::getShape() const{
    return shape_;
}

template <typename T>
T& Tensor<T>::operator()(std::vector<int> index){
    int i = Index(index);
    return (data_->data)[i];
}

template <typename T>
T& Tensor<T>::operator[](int index){
    return (data_->data)[index];
}

template <typename T>
void Tensor<T>::Set(std::vector<int> index, T value){
    int i = Index(index);
    (data_->data)[i] = value;
}

template <typename T>
void Tensor<T>::print(std::string str){
    Device device = data_->device;
    this->to("cpu");
    std::cout << str << std::endl;
    std::cout << "Tensor(";
    for (int i = 0; i < shape_.size(); i++){
        std::cout << shape_[i];
        if (i != shape_.size() - 1){
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl;
    std::cout << "data: [";
    int size = this->getSize();
    for (int i = 0; i < size; i++){
        std::cout << data_->data[i];
        if (i != size - 1){
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
    this->to(device);
}

template <typename T>
int Tensor<T>::Index(std::vector<int> indices) const{

    for (int i = 0; i < indices.size(); i++){
        if (indices[i] >= shape_[i]){
            std::cerr << "Error: index out of range" << std::endl;
            return -1;
        }
    }

    int index = 0;
    int stride = 1;
    for (int i = 0; i < indices.size(); i++){
        index += indices[i] * stride;
        stride *= shape_[i];
    }
    return index;
}

template <typename T>
Tensor<T> Tensor<T>::to(std::string device){
    if (device == "cuda"){
        return gpu();
    }
    else if (device == "cpu"){
        return cpu();
    }
    else{
        std::cerr << "Error: device not supported" << std::endl;
        exit(1);
    }
}

template <typename T>
Tensor<T> Tensor<T>::to(Device device){
    if (device == Device::CUDA){
        return gpu();
    }
    else if (device == Device::CPU){
        return cpu();
    }
    else{
        std::cerr << "Error: device not supported" << std::endl;
        exit(1);
    }
}

template <typename T>
Tensor<T> Tensor<T>::cpu(){
    this->data_->to(Device::CPU);
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::gpu(){
    this->data_->to(Device::CUDA);
    return *this;
}

template <typename T>
int Tensor<T>::getSize() const{
    return data_->size;
}

template <typename T>
Device Tensor<T>::getDevice() const{
    return data_->device;
}

// TODO

template <typename T>
void Tensor<T>::matmul(T alpha, const Tensor<T> &A, const Tensor<T> &B, T beta, Tensor<T> &C,bool transA,bool transB){

    // $ C = \alpha AB + \beta C $
    // 由于cublas的矩阵乘法要求A,B,C都是列优先的，可以利用如下公式：
    // $ C^T = \alpha B^T A^T + \beta C^T $

    static bool init = false;
    static cublasHandle_t handle;
    static cublasStatus_t status;
    if (!init){
        status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS){
            std::cerr << "Error: cublasCreate failed" << std::endl;
            exit(1);
        }
        init = true;
    }
    if (!std::is_same<T,float>::value && !std::is_same<T,double>::value){
        std::cerr << "Error: matmul only support float and double" << std::endl;
        exit(1);
    }
    if (A.getDevice() != Device::CUDA || B.getDevice() != Device::CUDA || C.getDevice() != Device::CUDA){
        std::cerr << "Error: matmul only support cuda tensor" << std::endl;
        exit(1);
    }
    if (A.getShape().size() != 2 || B.getShape().size() != 2 || C.getShape().size() != 2){
        std::cerr << "Error: matmul only support 2d tensor" << std::endl;
        exit(1);
    }
    auto A_shape = A.getShape();
    auto B_shape = B.getShape();
    auto C_shape = C.getShape();
    
    int aRow = transA?A_shape[1]:A_shape[0];
    int aCol = transA?A_shape[0]:A_shape[1];
    int bRow = transB?B_shape[1]:B_shape[0];
    int bCol = transB?B_shape[0]:B_shape[1];

    auto tA = transA?CUBLAS_OP_T:CUBLAS_OP_N;
    auto tB = transB?CUBLAS_OP_T:CUBLAS_OP_N;

    // 检查维度是否匹配
    // TODO


    cublasSgemm(
        handle,
        tB,
        tA,
        bCol,
        aRow,
        bRow,
        &alpha,
        B.data_->data,
        tB?bRow:bCol,
        A.data_->data,
        tA?aRow:aCol,
        &beta,
        C.data_->data,
        bCol
    );

}

template <typename T>
void Tensor<T>::reShape(std::vector<int> shape){
    int size = 1;
    for (int i = 0; i < shape.size(); i++){
        size *= shape[i];
    }
    if (size != this->getSize()){
        std::cerr << "Error: reshape size mismatch" << std::endl;
        exit(1);
    }
    shape_ = shape;
}

template <typename T>
void Tensor<T>::fill(T value){
    int size = this->getSize();
    if(getDevice()==Device::CPU){
        std::fill(data_->data,data_->data+size,value);
    }
    else{
        T* tmp = new T[size];
        std::fill(tmp,tmp+size,value);
        cudaMemcpy(data_->data,tmp,size*sizeof(T),cudaMemcpyHostToDevice);
        delete[] tmp;
    }
}

template <typename T>
void Tensor<T>::transpose(){
    if (shape_.size() != 2){
        std::cerr << "Error: transpose only support 2d tensor" << std::endl;
        exit(1);
    }
    for(int i=0;i<shape_[0];i++){
        for(int j=0;j<shape_[1];j++){
            std::swap((*this)({i,j}),(*this)({j,i}));
        }
    }
    std::swap(shape_[0],shape_[1]);
}

template <typename T>
Tensor<T> Tensor<T>::subtensor(const Tensor<T> &tensor,std::vector<int> shape,int offset){
    Tensor<T> result(shape,tensor.getDevice());
    result.data_->data = tensor.data_->data + offset;
    result.data_->sub = true;
    return result;
}

template <typename T>
void Tensor<T>::randomfill(){
    // 使用curand随机填充
    if(getDevice()==Device::CPU){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(0,1);
        for(int i=0;i<getSize();i++){
            data_->data[i] = dis(gen);
        }
    }
    else{
        curandGenerator_t gen;
        curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
        curandGenerateUniform(gen,data_->data,getSize());
        curandDestroyGenerator(gen);
    }
}


#endif // TENSOR_H
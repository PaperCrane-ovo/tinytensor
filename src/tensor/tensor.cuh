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
    Tensor(std::vector<int> shape, std::string device = "cpu");
    Tensor(std::vector<int> shape,Device device=Device::CPU);
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
    void print();

    Tensor to(std::string device);
    Tensor to(Device device);
    Tensor cpu();
    Tensor gpu();

    int getSize() const;
    Device getDevice() const;
    std::shared_ptr<DataStorage<T>> data_;

private:
    std::vector<int> shape_;

    

    int Index(std::vector<int> indices) const;
};



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
    data_ = std::make_shared<DataStorage<T>>(s,Device::CPU);
    std::memmove(data_->data, a, s * sizeof(T));
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
void Tensor<T>::print(){
    Device device = data_->device;
    this->to("cpu");
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


#endif // TENSOR_H
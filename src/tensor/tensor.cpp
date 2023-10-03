#include "tensor.hpp"



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
//TODO: implement copy constructor
template<typename T>
Tensor<T>::Tensor(const Tensor<T>& tensor){
    shape_ = tensor.shape_;
    data_ = tensor.data_;
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
void Tensor<T>::Set(std::vector<int> index, T value){
    int i = Index(index);
    (data_->data)[i] = value;
}

/* template <typename T>
void Tensor<T>::print(){
    std::cout << "Tensor(";
    for (int i = 0; i < shape_.size(); i++){
        std::cout << shape_[i];
        if (i != shape_.size() - 1){
            std::cout << ", ";
        }
    }
    std::cout << ")" << std::endl;
    std::cout << "data: [";
    for (int i = 0; i < size_; i++){
        std::cout << data_[i];
        if (i != size_ - 1){
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
} */

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
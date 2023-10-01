#include "tensor.hpp"

using namespace Crane;

template <typename T>
Tensor<T>::Tensor(std::vector<int> shape):Tensor<T>::Tensor(shape,"cpu"){
}


template<typename T>
Tensor<T>::Tensor(std::vector<int> shape,std::string device){
    device_ = device == "cpu" ? Device::CPU : Device::CUDA;
    shape_ = shape;
    size_ = 1;
    for (int i = 0; i < shape.size(); i++){
        size_ *= shape[i];
    }
    switch(device_){
        case Device::CPU:
            data_ = std::make_shared<T>(new T[size_],deleteData);
            break;
        case Device::CUDA:
            data_ = std::shared_ptr<T>(nullptr,deleteData);
            cudaMalloc(data_.get(),size_ * sizeof(T));
            break;
        default:
            std::cout << "Error: device not supported" << std::endl;
    }
}
//TODO: implement copy constructor
template<typename T>
Tensor<T>::Tensor(const Tensor<T>& tensor){
    device_ = tensor.device_;
    shape_ = tensor.shape_;
    size_ = tensor.size_;
    data_ = tensor.data_;
}

template <typename T>
void deleteData(T* ptr){
    switch(device_){
        case Device::CPU:
            delete[] ptr;
            break;
        case Device::CUDA:
            cudaFree(ptr);
            break;
        default:
            std::cout << "Error: device not supported" << std::endl;
    }
}

template <typename T>
Tensor<T>::~Tensor(){
    if (data_ == nullptr){
        return;
    }
    switch (device_){
        case Device::CPU:
            delete[] data_;
            break;
        case Device::CUDA:
            cudaFree(data_);
            break;
        default:
            std::cout << "Error: device not supported" << std::endl;
    }
}



template <typename T>
std::vector<int> Tensor<T>::getShape() const{
    return shape_;
}

template <typename T>
T& Tensor<T>::operator()(std::vector<int> index){
    int i = Index(index);
    return data_[i];
}

template <typename T>
void Tensor<T>::Set(std::vector<int> index, T value){
    int i = Index(index);
    data_[i] = value;
}

template <typename T>
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
}

template <typename T>
int Tensor<T>::Index(std::vector<int> indices) const{
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
        std::cout << "Error: device not supported" << std::endl;
    }
}

template <typename T>
Tensor<T> Tensor<T>::cpu(){
    if(device_ == Device::CPU){
        return this;
    }
    else{
        T* data_cpu = new T[size_];
        cudaMemcpy(data_cpu,this->data_,size_ * sizeof(T), cudaMemcpyDeviceToHost);
        cudaFree(this->data_);
        this->data_ = data_cpu;
        this->device_ = Device::CUDA;
        return this;
    }
}

template <typename T>
Tensor<T> Tensor<T>::gpu(){
    if(device_ == Device::CUDA){
        return this;
    }
    else{
        T* data_gpu;
        cudaMalloc(&data_gpu,size_ * sizeof(T));
        cudaMemcpy(data_gpu,this->data_,this->size_ * sizeof(T), cudaMemcpyHostToDevice);
        delete[] this->data_;
        this->data_ = data_gpu;
        this->device_ = Device::CUDA;
        return this;
    }
}
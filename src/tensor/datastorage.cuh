#ifndef DATASTORAGE_HPP
#define DATASTORAGE_HPP

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include "../cuda/cudautils.cuh"


template <typename T>
class DataStorage{
    public:
        DataStorage(int size, Device device);
        DataStorage(T* data,int size, Device device);

        ~DataStorage();
        void to(std::string device);
        void to(Device device);

        T* data;
        int size;
        Device device;


};

template <typename T>
DataStorage<T>::DataStorage(int size,Device device) {
    this->size = size;
    this->device = device;
    if (device == Device::CPU) {
        this->data = new T[size];
    } else {
        cudaMalloc(&this->data,size * sizeof(T));
    }
}
template <typename T>
DataStorage<T>::DataStorage(T* data,int size,Device device){
    this->size = size;
    this->device = device;
    if (device == Device::CPU) {
        this->data = new T[size];
        memcpy(this->data,data,size * sizeof(T));
    } else {
        cudaMalloc(&(this->data),size * sizeof(T));
        cudaMemcpy(this->data,data,size * sizeof(T),cudaMemcpyHostToDevice);
    }
}
template <typename T>
DataStorage<T>::~DataStorage() {
    if (device == Device::CPU) {
        delete[] data;
    } else {
        cudaFree(data);
    }
}
template <typename T>
void DataStorage<T>::to(std::string device){
    if (device == "cpu")
        this->to(Device::CPU);
    else if (device == "cuda")
        this->to(Device::CUDA);
    else
        std::cerr<<"Invalid device"<<std::endl;
}
template <typename T>
void DataStorage<T>::to(Device device){
    switch (device){
        case Device::CPU:
            if (this->device == Device::CUDA) {
                T* temp = new T[size];
                cudaMemcpy(temp,this->data,size * sizeof(T),cudaMemcpyDeviceToHost);
                cudaFree(this->data);
                this->data = temp;
                this->device = Device::CPU;
            }
            break;
        case Device::CUDA:
            if (this->device == Device::CPU) {
                T* temp;
                cudaMalloc(&temp,size * sizeof(T));
                cudaMemcpy(temp,this->data,size * sizeof(T),cudaMemcpyHostToDevice);
                delete[] this->data;
                this->data = temp;
                this->device = Device::CUDA;
            }
            break;
        default:
            std::cerr<<"Invalid device"<<std::endl;

    }
}

#endif // DATASTORAGE_HPP
#ifndef DATASTORAGE_HPP
#define DATASTORAGE_HPP

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include "../cuda/cudautils.hpp"


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

#endif // DATASTORAGE_HPP
#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include "../cuda/cudautils.hpp"
#include "datastorage.hpp"
#include "datastorage.cu"
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
    Tensor cpu();
    Tensor gpu();

    int getSize() const;
    Device getDevice() const;


private:
    std::vector<int> shape_;

    std::shared_ptr<DataStorage<T>> data_;

    int Index(std::vector<int> indices) const;
};


#endif // TENSOR_H
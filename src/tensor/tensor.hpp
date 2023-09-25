#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <string>
#include <memory>


namespace Crane
{
    template <typename T>
    class Tensor{
        public:
            Tensor(std::vector<int> shape);
            Tensor (std::vector<int> shape, std::string device);
            Tensor(const Tensor& tensor);
            ~Tensor();
            // 获取张量的形状
            std::vector<int> getShape() const;
            // 获取张量的数据
            T& operator()(std::vector<int> index);
            
            void Set(std::vector<int> index, T value);
            void print();

            Tensor to(std::string device);
            Tensor cpu();
            Tensor gpu();
            
            

        private:
            std::vector<int> shape_;
            std::shared_ptr<T> data_;
            void deleteData(T* ptr);
            int size_;
            std::string device_;

            int Index(std::vector<int> indices) const;



    };
}

#endif // TENSOR_H
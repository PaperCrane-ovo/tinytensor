#include "tensor/tensor.cuh"
#include "modules/modules.cuh"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using Tensorf = Tensor<float>;
using Modulef = Module<float>;
using Reluf = ReLU<float>;
using Sigmoidf = Sigmoid<float>;
using Softmaxf = SoftMax<float>;
using Linearf = FullyConnected<float>;
using Convf = Convolution<float>;
using MaxPoolf = MaxPooling<float>;
using CrossEntropyLossf = CrossEntropyLoss<float>;

PYBIND11_MODULE(Tensor,m){
    py::enum_<Device> device(m,"Device");

    py::class_<Tensorf>(m,"Tensorf")
        .def(py::init<>)
        .def(py::init<const std::vector<uint32_t>,Device>())
        .def(py::init<const std::vector<uint32_t>,std::string>())
        .def(py::init<const std::vector<uint32_t>,Device,std::vector<float>>())
        .def("get_shape",&Tensorf::getShape)
        .def("set",&Tensorf::Set)
        .def("print",&Tensorf::print)
        .def("to", static_cast<void(Tensor::*)(std::string)>(&Tensor::to))
        .def("to", static_cast<void(Tensor::*)(Device)>(&Tensor::to))

        .def("cpu",&Tensorf::cpu)
        .def("gpu",&Tensorf::gpu)
        .def("get_size",&Tensorf::getSize)
        .def("get_device",&Tensorf::getDevice)
        .def_static("matmul",&Tensorf::matmul)
        .def("reshape",&Tensorf::reShape)
        .def("fill",&Tensorf::fill)
        .def("transpose",&Tensorf::transpose)
        .def("randomfill",&Tensorf::randomfill)
        .def_static("sub_tensor",&Tensorf::subtensor);

    py::class_<Modulef>(m,"Modulef")
        .def(py::init<>())
        .def("get_weights_grad",&Modulef::getWeightGrad)
        .def("get_bias_grad",&Modulef::getBiasGrad);

    py::class_<Reluf>(m,"Reluf",py::base<Modulef>())
        .def(py::init<>())
        .def("forward",&Reluf::forward)
        .def("backward",&Reluf::backward);

    py::class_<Sigmoidf>(m,"Sigmoidf",py::base<Modulef>())
        .def(py::init<>())
        .def("forward",&Sigmoidf::forward)
        .def("backward",&Sigmoidf::backward);
    
    py::class_<Linearf>(m,"Linearf",py::base<Modulef>())
        .def(py::init<uint32_t,uint32_t>())
        .def(py::init<const Tensorf&,const Tensorf&>())
        .def("forward",&Softmaxf::forward)
        .def("backward",&Softmaxf::backward);

    py::class_<Convf>(m,"Convf",py::base<Modulef>())
        .def(py::init<uint32_t,uint32_t>())
        .def(py::init<const Tensorf&>())
        .def(py::init<const Tensorf&,const Tensorf&>())
        .def("forward",&Convf::forward)
        .def("backward",&Convf::backward);

    py::class_<MaxPoolf>(m,"MaxPoolf",py::base<Modulef>())
        .def("forward",&MaxPoolf::forward)
        .def("backward",&MaxPoolf::backward);

    py::class_<Softmaxf>(m,"Softmaxf",py::base<Modulef>())
        .def("forward",&Softmaxf::forward)
        .def("backward",&Softmaxf::backward);

    py::class_<CrossEntropyLossf>(m,"CrossEntropyLossf",py::base<Modulef>())
        .def(py::init<>())
        .def(py::init<const Tensorf&>())
        .def(py::init<uint32_t,uint32_t>())
        .def("forward",&CrossEntropyLossf::forward)
        .def("backward",&CrossEntropyLossf::backward);
}
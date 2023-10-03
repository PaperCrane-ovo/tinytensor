# TinyTensor
## 目标:实现一个支持 gpu 的深度学习框架.
## Dependencies
- ~~**Eigen**:一个矩阵库,可用于线性代数运算.~~ 目前已经不需要。
- **待续**
## TODO-LIST
- Tensor
- 封装
- 计算图,反向传播...
- 加速

## 项目组织
使用 CMake 作为项目依赖管理.

## Tensor实现:
使用1维数组实现。目前 tensor 类的封装工作已经做好，索引已经完成，实现了在 gpu 和 cpu 之间的操作。暂时不支持张量之间的运算以及广播机制（wip）。

为了方便内存管理，使用了智能指针 `shared_ptr` 来管理一个 `DataStorage` 对象，这个对象封装了数组指针和设备，利用对象的销毁自动释放内存（RAII）





## 激活函数：
目前实现了 `ReLU` 和 `Sigmoid` 函数的正向及反向传播，同时支持 `cpu` 和 `gpu` 上的运算。

如果需要自定义激活函数，需要继承`ActivationFunction`基类并实现`forward`、`backward`、`getName`函数，并尽量实现核函数，以防不在同一个 device 上出现异常。
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"
sources = ["src/pybind.cu"]
setup(
    name="tinytensor",
    version=__version__,
    author="papercrane",
    author_email="2100013134@stu.pku.edu.cn",
    packages=find_packages(),
    zip_safe=False,
    install_requires=["torch"],
    python_requires=">=3.8",
    license="MIT",
    ext_modules=[
        CUDAExtension(
            name="tinytensor", 
            sources=sources,
            include_dirs=['include','third_party/pybind11/include'],
            libraries=['cublas','curand']
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
)

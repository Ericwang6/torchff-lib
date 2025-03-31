import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if 'CXX' not in os.environ:
    os.environ['CXX'] = 'g++'
    print("Set C++ compiler: g++")


setup(
    name='torchff',
    classifiers=[
        'Development Status :: Beta',
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.12',
    ],
    packages=find_packages(exclude=['csrc', 'tests', 'docs']),
    ext_modules=[
        CUDAExtension(
            name='torchff_harmonic_bond',
            sources=['csrc/bond/harmonic_bond_interface.cpp', 'csrc/bond/harmonic_bond_cpu.cpp', 'csrc/bond/harmonic_bond_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_80']
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

# Developer Note

## How to write a customized PyTorch OP?

To implement a customized PyTorch OP in C++/CUDA, there needs to be three parts:

1. C++/CUDA Source Files

+ `csrc/bond/harmonic_bond_cpu.cpp`

+ `csrc/bond/harmonic_bond_cuda.cu`

+ `csrc/bond/harmonic_bond_interface.cpp`


2. Python Wrappers

+ `torchff/bond.py`

3. Build Scrpit
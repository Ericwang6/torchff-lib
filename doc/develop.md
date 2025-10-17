# Developer Note

## How to develop a customized operator in PyTorch?

1. Write CUDA Source Files

+ Implemented `forward` and `backward` calculation in a child class of `torch::autograd::Function` if you use your own cuda kernels. For example, in `csrc/bond/harmonic_bond_cuda.cu`, the harmonic bond calculations are defined in `HarmonicBondFunctionCuda` class. If your customized operator is implemented in pure torch functions, you can skip this step and directly define your calculations in a C++ function.

+ Wrap the calculation in a C++ function. For example, `compute_harmonic_bond_energy_cuda` in the same file.

+ Register this function as an implementation of the operator named `compute_harmonic_bond_energy` under namespace `torchff`:

```C
TORCH_LIBRARY_IMPL(torchff, AutogradCUDA, m) {
    m.impl("compute_harmonic_bond_energy", compute_harmonic_bond_energy_cuda);
}
```

2. Write a C++ interface file to register this operator and define its schema. Fo rexample, `csrc/bond/harmonic_bond_interface.cpp` defines:

```C
#include <pybind11/pybind11.h>
#include <torch/library.h>
#include <torch/extension.h>

TORCH_LIBRARY_FRAGMENT(torchff, m) {
    m.def("compute_harmonic_bond_energy(Tensor coords, Tensor pairs, Tensor b0, Tensor k) -> Tensor");
    m.def("compute_harmonic_bond_forces(Tensor coords, Tensor pairs, Tensor b0, Tensor k, Tensor (a!) forces) -> ()");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "torchff harmonic bond CUDA extension";
}
```

3. Call this operator in python
```python
import torch
# Because we bind the C++ function to the module named `TORCH_EXTENSION_NAME` in the interface file,
# we need to import this module to find the implementation of this operator
# In torchff-lib, `TORCH_EXTENSION_NAME` is a macro defined during compilation and it follows the 
# naming convention of `torchff_{NAME}` where {NAME} is the name of the directory where the interface file
# is located in the source code. Check `setup.py` for more details
import torchff_bond

# The operators is called via `torch.ops.torchff`, where `torchff` is the naming space defined
# in `TORCH_LIBRARY_FRAGMENT`
torch.ops.torchff.compute_harmonic_bond_energy(coords, bonds, b0, k)
```

## References

+ https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html
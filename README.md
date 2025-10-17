# torchff-lib
A Pytorch library to implement force field terms

## Environment Setup
+ python == 3.12
+ pytorch >= 2.4.0
+ cuda >= 12.4
+ pytest

For example, ere is a code snippet to setup environment with mamba and pip
```bash
mamba create -n {ENV_NAME} python=3.12
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install pytest
pip install dependency
```

## Installation

```bash
mamba activate {ENV_NAME}
git clone https://github.com/Ericwang6/torchff-lib.git
cd torchff-lib
python setup.py install
```

## Developer Guide

+ [A brief introduction to write customized operators in torchff](doc/develop.md)

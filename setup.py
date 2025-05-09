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
            sources=[
                'csrc/bond/harmonic_bond_interface.cpp',
                'csrc/bond/harmonic_bond_cpu.cpp',
                'csrc/bond/harmonic_bond_cuda.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_80']
            },
            include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")]
        ),
        CUDAExtension(
            name='torchff_harmonic_angle',
            sources=[
                'csrc/angle/harmonic_angle_interface.cpp',
                'csrc/angle/harmonic_angle_cpu.cpp',
                'csrc/angle/harmonic_angle_cuda.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_80']
            },
            include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")]
        ),
        CUDAExtension(
            name='torchff_coulomb',
            sources=['csrc/coulomb/coulomb_interface.cpp', 'csrc/coulomb/coulomb_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_80']
            },
            include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")]
        ),
        CUDAExtension(
              name='torchff_periodic_torsion',
              sources=['csrc/torsion/periodic_torsion_interface.cpp', 'csrc/torsion/periodic_torsion_cuda.cu'],
              extra_compile_args={
                  'cxx': ['-O3'],
                  'nvcc': ['-O3', '-arch=sm_80']
              },
              include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")]
          ),
        CUDAExtension(
            name='torchff_vdw',
            sources=['csrc/vdw/lennard_jones_interface.cpp', 'csrc/vdw/lennard_jones_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_80']
            },
            include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")]
        ),
        CUDAExtension(
            name='torchff_nblist',
            sources=[
                'csrc/nblist/nblist_interface.cpp', 
                'csrc/nblist/nblist_nsquared_cuda.cu', 
                'csrc/nblist/nblist_clist_cuda.cu',
                'csrc/nblist/nblist_cluster_pairs_cuda.cu'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_80']
            },
            include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")]
        ),
        CUDAExtension(
            name='torchff_nb',
            sources=[
                'csrc/nonbonded/nonbonded_interface.cpp',
                'csrc/nonbonded/nonbonded_atom_pairs_cuda.cu',  
                'csrc/nonbonded/nonbonded_cuda.cu', 
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-arch=sm_80']
            },
            include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

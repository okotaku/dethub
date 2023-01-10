from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='ops',
    ext_modules=[
        CUDAExtension(
            name='ops',
            sources=[
                'pybind.cpp',
                'check_prior_in_gt_kernel.cu',
                'check_prior_in_gt.cpp',
                'check_prior_in_gt_kernel_dsla.cu',
                'check_prior_in_gt_dsla.cpp',
                'binary_cross_entropy_cost_kernel.cu',
                'binary_cross_entropy_cost.cpp',
            ],
            extra_compile_args={'nvcc': ['-O3']})
    ],
    cmdclass={'build_ext': BuildExtension})

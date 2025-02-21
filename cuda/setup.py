from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='neighborhood_aggregation',
    ext_modules=[
        CUDAExtension(
            name='neighborhood_aggregation',
            sources=['neighborhood_aggregation.cu'],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)

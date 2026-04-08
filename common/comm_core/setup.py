import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, _find_cuda_home

os.system('pip uninstall -y comm_core')

CUDA_DIR = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

if CUDA_DIR is None:
    CUDA_DIR = _find_cuda_home()
print('CUDA_DIR: ', CUDA_DIR)

NCCL_DIR = '/data/home/sczd744/.conda/envs/py38-hvd'
NCCL_DIR =os.environ.get('NCCL_HOME', NCCL_DIR) 
MPI_DIR = '/data/home/sczd744/.conda/envs/py38-hvd'
MPI_DIR =os.environ.get('MPI_HOME', MPI_DIR) 

# Python interface
setup(
    name='comm_core',
    version='0.1.1',
    install_requires=['torch'],
    packages=['comm_core'],
    package_dir={'comm_core': './'},
    ext_modules=[
        CUDAExtension(
            name='comm_core',
            include_dirs=['./', 
                NCCL_DIR+'/include', 
                MPI_DIR+'/include',
                CUDA_DIR+'/samples/common/inc'],
            sources=[
                'src/communicator.cpp',
                'src/comm_core.cpp',
            ],
            libraries=['nccl', 'mpi'], # 删去了'mpi_cxx'，应为openmpi4.1.1不用这个了。
            library_dirs=['objs', CUDA_DIR+'/lib64', NCCL_DIR+'/lib', MPI_DIR+'/lib'],
            extra_link_args=['-Wl,-rpath,' + MPI_DIR + '/lib'],  # 可选：嵌入运行时路径，防止加载失败
            # extra_compile_args=['-g']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Shaohuai Shi',
    author_email='shaohuais@cse.ust.hk',
    description='Efficient PyTorch Extension for Communication based on NCCL',
    keywords='Pytorch C++ Extension',
    zip_safe=False,
)

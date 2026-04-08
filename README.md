# DeAR: <u>de</u>coupling the <u>a</u>ll-<u>r</u>educe primitive to accelerate distributed deep learning

## Introduction 
We propose a new optimization algorithm called DeAR, that decouples the all-reduce primitive to two operations, so as to enable fine-grained scheduling without introducing extra communication overhead. This repository contains DeAR's source code, as well as a set of benchmarking scripts for evaluating the training performance of popular distributed deep learning methods with data parallelism. Currently, it covers: 
### Optimization algorithms without Tensor Fusion
- Wait-free backpropagation (WFBP), which is also known as the technique of pipelining the backward computations with gradient communications. 
- [ByteScheduler](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler), which uses tensor partition and priority schedule to overlap some communication tasks with
feed-forward computing tasks. 
- DeAR w/o TF, which disables the tensor fusion technique by setting THRESHOLD=None and NUM_NEARBY_LAYERS=1. 
### Optimization algorithms with Tensor Fusion
- [Horovod](https://github.com/horovod/horovod). 
- [PyTorch-DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
- [MG-WFBP](https://github.com/HKBU-HPML/MG-WFBP), which determines fusion tensors by measuring the backward computation time and communication time. 
- DeAR, which supports tuning tensor fusion with [Bayesian optimization](https://github.com/fmfn/BayesianOptimization). 

### Deep Neural Networks
- [Convolutional neural networks (CNNs)](https://pytorch.org/vision/stable/models.html) on a fake ImageNet data set (i.e., randomly generate the input image of 224\*224\*3)
- [Transformers](https://github.com/huggingface/transformers): BERT-Base and BERT-Large pretraining models.

## Installation
### Prerequisites
- Python 3.6+
- CUDA-10.+
- NCCL-2.4.+
- [PyTorch-1.8.+](https://download.pytorch.org/whl/torch_stable.html)
- [OpenMPI-4.0.+](https://www.open-mpi.org/software/ompi/v4.0/)
- [Horovod-0.19.+](https://github.com/horovod/horovod)
- [ByteScheduler](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler)

### 前置条件：
- rust编辑器：（once）
RUSTUP_UPDATE_ROOT=https://mirrors.tuna.tsinghua.edu.cn/rustup/rustup \
> RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup \
> sh -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh
- 虚拟环境：
conda activate py38-hvd
- 启动modules功能（并行云专属：）
source /etc/profile.d/modules.sh 
- cmake:
module load cmake/3.30
- 加载mpi,加载gcc，加载CUDA
# module load openmpi/4.1.5_gcc11.4_ucx1.14.1_cuda11.8
module load openmpi/4.1.1 （暂时不用）
module load cuda/11.8 （暂时不用）
module load nccl/2.16.5_cu118 （暂时不用）

### Get the code
```
$git clone https://github.com/lzhangbv/dear_pytorch.git
$cd dear_pytorch
下载库用下面两条：

$pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
+
pip install -r requirements.txt


￥修改：
/data/home/sczd744/.conda/envs/py38-hvd/bin/pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \                                       
 --index-url https://download.pytorch.org/whl/cu118
 
/data/home/sczd744/.conda/envs/py38-hvd/bin/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --trusted-host pypi.tuna.tsinghua.edu.cn（避免python12抢先）

python -m pip install \  (pip 可能搞错，但是)
torch==2.1.2+cu118 \
torchvision==0.16.2+cu118 \
torchaudio==2.1.2+cu118 \
--index-url https://download.pytorch.org/whl/cu118


￥ 修改：----------------------------------------------------------------帮助horovod的安装----------------------------------------------------------------
# ===== 加载模块 ===== 改用conda下载horovod之后，GPT说不需要再用module加载下面2个库了。
module load cmake/3.30 
module load cuda/11.8  
module load openmpi/4.1.1 （暂时不用）
module load nccl/2.16.5_cu118 

# ===== 编译器设置 =====
# C++17 设置（多重保险）
export CC=gcc
export CXX=g++
export CXXFLAGS="-std=c++17 -fPIC"
export CFLAGS="-fPIC"
export CMAKE_CXX_STANDARD=17
export CMAKE_CXX_STANDARD_REQUIRED=ON
export CMAKE_CXX_FLAGS="-std=c++17"

# 强制 CMake 使用 C++17
export HOROVOD_CMAKE_ARGS="-DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON"

# ===== Horovod 组件选择 =====
export HOROVOD_WITH_MPI=1
export HOROVOD_WITH_PYTORCH=1
export HOROVOD_WITH_TENSORFLOW=0
export HOROVOD_WITH_MXNET=0

# ===== GPU 支持 =====
export HOROVOD_GPU_OPERATIONS=NCCL
export HOROVOD_NCCL_HOME=/data/apps/nccl/2.16.5-1+cuda11.8
export HOROVOD_NCCL_LINK=SHARED

# （可选，避免 CMake 自动探测）
export HOROVOD_CMAKE_ARGS="-DHOROVOD_WITH_TENSORFLOW=OFF"

# 验证
echo "=== Checking NCCL ==="
if [ -f "$HOROVOD_NCCL_HOME/lib/libnccl.so" ]; then
    echo "✅ NCCL found at $HOROVOD_NCCL_HOME"
else
    echo "❌ NCCL not found!"
    exit 1
fi

# 显示配置
echo ""
echo "=== Configuration ==="
env | grep HOROVOD | sort

# 安装

# ===== 安装 =====原本是horovod==0.21.3
python -m pip install horovod==0.28.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

export CC=$(which gcc)
export CXX=$(which g++)

HOROVOD_WITH_PYTORCH=1 \
HOROVOD_WITH_TENSORFLOW=0 \
HOROVOD_GPU_OPERATIONS=NCCL \
HOROVOD_CMAKE=/data/apps/cmake/3.30/bin/cmake \
HOROVOD_NCCL_HOME=/data/apps/modulefiles/nccl/2.16.5_cu118 \
HOROVOD_CUDA_HOME=/data/apps/cuda/11.8 \
HOROVOD_NCCL_HOME=/data/apps/nccl/2.16.5-1+cuda11.8 \
HOROVOD_NCCL_INCLUDE=/data/apps/nccl/2.16.5-1+cuda11.8/include \
HOROVOD_NCCL_LIB=/data/apps/nccl/2.16.5-1+cuda11.8/lib/libnccl.so \
pip install --no-cache-dir horovod==0.28.1

```
(新的GPT不推荐)Claude 推荐用 conda来下，更快： conda install -n py38-hvd -c conda-forge horovod=0.28.1 （很不错！包干了）或 /data/apps/miniforge3/envs/py38/bin/conda install -c conda-forge horovod=0.28.1


If pip installation failed, please try to upgrade pip via `pip install --upgrade pip`. If Horovod installation with NCCL failed, please check the installation [guide](https://horovod.readthedocs.io/en/stable/install_include.html). To run ByteScheduler, please check the installation [instruction](https://github.com/bytedance/byteps/tree/bytescheduler/bytescheduler) and it was found to be compatible with PyTorch 1.4. 

If you have encountered other errors during installation, please check the [install document](https://github.com/lzhangbv/dear_pytorch/blob/master/install.md) (contributed by Haoxuan Yu), and we recommend using the same software versions according to our [paper](https://arxiv.org/pdf/2302.12445.pdf) (section VI.A). 

### Configure the cluster settings
Before running the scripts, please carefully configure the configuration files in the directory of `configs`.
- configs/cluster\*: configure the host files for MPI
- configs/envs.conf: configure the cluster environments
# 
在运行脚本之前，请仔细配置`configs`目录中的配置文件。
- configs/cluster\*：配置MPI的主机文件
- configs/envs.conf：配置集群环境

Compile the communication package:
```
$ cd common/comm_core
$ bash compile.sh
```

Create a log folder in the dear_pytorch dir, e.g., 
```
$mkdir -p logs/sc22-tf
```

### Run benchmarks
- The batch mode
```
$python benchmarks.py
```
# 针对不同的实验设置，用户可在benckmarks.py脚本中修改深度神经网络模型、批处理大小、GPU数量及网络配置参数。
For different experimental settings, users can modify the DNN model, batch size, the number of GPUs, and network configurations in the benckmarks.py script. 


- The individual mode, e.g.,
```
$cd dear
$dnn=resnet50 bs=64 nworkers=64 ./horovod_mpi_cj.sh
```

Before running DeAR w/o tensor fusion, please set THRESHOLD=None and NUM_NEARBY_LAYERS=1 in the DeAR's dopt_rsag.py script. For DeAR with tensor fusion, we use THRESHOLD=25MB by default. To support Bayesian optimization, please import dopt_rsag_bo and increase the num-warmup-batches to at least 60 to tune buffer size in DeAR's benchmark scripts. 
在运行未启用张量融合的DeAR之前，请在DeAR的dopt_rsag.py脚本中设置THRESHOLD=None和NUM_NEARBY_LAYERS=1。对于启用张量融合的DeAR，默认采用THRESHOLD=25MB。为支持贝叶斯优化，请导入dopt_rsag_bo脚本，并将基准测试脚本中的num-warmup-batches参数增至至少60以调整缓冲区大小。
## DeAR Usage
The DeAR distributed optimizer can be easily used like `horovod.DistributedOptimizer()`.
```Python
import dear
dear.init()
... 
optimizer = optim.SGD(model.parameters(), ...)
optimizer = dear.DistributedOptimizer(optimizer, ...)
... 
for i, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
...
```

### DeAR Example
Example script for training on MNIST was provided.
```
$ bash mnist.sh
```

## Paper
If you are using this repository for your paper, please cite our work
```
@article{zhang2023decoupling,
  title={Decoupling the All-Reduce Primitive for Accelerating Distributed Deep Learning},
  author={Zhang, Lin and Shi, Shaohuai and Chu, Xiaowen and Wang, Wei and Li, Bo and Liu, Chengjian},
  journal={arXiv preprint arXiv:2302.12445},
  year={2023}
}
```

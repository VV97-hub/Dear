#!/bin/bash
# Read-only environment probe for the Helix camera-ready paper.
# No training or benchmark is launched.

set +e

section() {
    echo
    echo "===== $1 ====="
}

section "JOB"
date --iso-8601=seconds
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

section "OS_KERNEL"
cat /etc/os-release
uname -a

section "CPU"
lscpu

section "MEMORY"
free -h

section "GPU_QUERY"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,pci.bus_id,pci.link.gen.current,pci.link.gen.max,pci.link.width.current,pci.link.width.max,power.limit --format=csv,noheader

section "GPU_TOPOLOGY"
nvidia-smi topo -m

section "GPU_FULL_RELEVANT"
nvidia-smi -q | grep -E "Product Name|VBIOS Version|CUDA Version|Total[[:space:]]*:|Current Link Gen|Max Link Gen|Current Link Width|Max Link Width|BAR1 Memory Usage|Power Limit" | head -100

section "PCI_DEVICES"
lspci -nn | grep -Ei "NVIDIA|Mellanox|Ethernet|Network"

section "NETWORK_LINKS"
ip -br link
ip -br addr

section "BOND0"
cat /proc/net/bonding/bond0 2>/dev/null
ethtool bond0 2>/dev/null | grep -E "Speed:|Duplex:|Port:|Link detected:"

section "RDMA"
command -v rdma && rdma link
command -v ibv_devices && ibv_devices
command -v ibv_devinfo && ibv_devinfo
command -v ibstat && ibstat

section "SOFTWARE_BASE"
gcc --version | head -1
python3 --version

source /data/apps/miniforge3/etc/profile.d/conda.sh
conda activate py38-hvd
source /etc/profile.d/modules.sh 2>/dev/null
module load cuda/11.8 2>/dev/null
module load gcc/11.4.0 2>/dev/null
module load openmpi/4.1.5_gcc11.4_ucx1.14.1_cuda11.8 2>/dev/null

section "CUDA_MPI_COMPILER"
which python3
python3 --version
nvcc --version
gcc --version | head -1
mpirun --version | head -3

section "PYTHON_STACK"
python3 - <<'PY'
import sys
print("python_executable=", sys.executable)
import torch
print("torch_version=", torch.__version__)
print("torch_cuda_build=", torch.version.cuda)
print("cudnn_version=", torch.backends.cudnn.version())
print("nccl_version=", torch.cuda.nccl.version())
print("cuda_available=", torch.cuda.is_available())
print("gpu_name=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
try:
    import horovod
    print("horovod_version=", getattr(horovod, "__version__", "unknown"))
except Exception as exc:
    print("horovod_import_error=", repr(exc))
try:
    import comm_core
    print("comm_core_path=", comm_core.__file__)
    print("comm_core_has_timeline_range_api=", hasattr(comm_core.Communicator, "syncEventElapsedRangeFromBase"))
except Exception as exc:
    print("comm_core_import_error=", repr(exc))
PY

section "HELIX_NCCL_SETTINGS"
echo "NCCL_IB_DISABLE=0"
echo "NCCL_IB_HCA=mlx5_bond_0"
echo "NCCL_SOCKET_IFNAME=bond0"
echo "NCCL_IB_GID_INDEX=3"
echo "NCCL_LAUNCH_MODE=PARALLEL (reported by prior NCCL runtime logs)"

section "DONE"
date --iso-8601=seconds

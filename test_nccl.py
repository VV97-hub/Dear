import os
import torch
import torch.distributed as dist

rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

# 获取PyTorch的库路径
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
print(f"PyTorch库路径: {torch_lib_path}")

# print(f"PyTorch版本: {torch.__version__}")
# print(f"PyTorch使用的NCCL版本: {torch.cuda.nccl.version()}")
# print(f"NCCL是否可用: {torch.distributed.is_nccl_available()}")
# print(f"CUDA是否可用: {torch.cuda.is_available()}")
#print(f"可用GPU数量: {torch.cuda.device_count()}")
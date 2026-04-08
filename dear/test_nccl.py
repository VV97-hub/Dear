import os, torch

print(
    "PID", os.getpid(),
    "OMPI_RANK", os.environ.get("OMPI_COMM_WORLD_RANK"),
    "LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"),
    "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"),
    "current_device", torch.cuda.current_device() if torch.cuda.is_available() else None,
    flush=True
)

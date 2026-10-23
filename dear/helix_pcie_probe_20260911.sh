#!/bin/bash
set +e
echo "===== GPU_BASIC ====="
nvidia-smi --query-gpu=name,pci.bus_id,driver_version,memory.total,power.limit --format=csv,noheader
echo "===== NVIDIA_PCI_DOMAIN ====="
nvidia-smi -q -d PCI
echo "===== LSPCI_LINK ====="
BDF=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | sed 's/^00000000://')
echo "BDF=$BDF"
lspci -s "$BDF" -vv | grep -E "LnkCap:|LnkSta:|NUMA node"
echo "===== SYSFS_LINK ====="
SYS_BDF=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | sed 's/^00000000:/0000:/')
DEV="/sys/bus/pci/devices/$SYS_BDF"
echo "SYS_BDF=$SYS_BDF"
for item in current_link_speed max_link_speed current_link_width max_link_width numa_node; do
    printf "%s=" "$item"
    cat "$DEV/$item" 2>/dev/null || echo "unavailable"
done

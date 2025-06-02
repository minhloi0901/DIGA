#!/bin/bash

echo "=== Host Shared Memory Status ==="
echo "Current /dev/shm size on host:"
df -h /dev/shm

echo -e "\n=== Host Shared Memory Usage ==="
ipcs -m

echo -e "\n=== Container Shared Memory Status ==="
echo "Current /dev/shm size in container:"
df -h /dev/shm

echo -e "\n=== Container Memory Info ==="
free -h

echo -e "\n=== Container Memory Limits ==="
if [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
    cat /sys/fs/cgroup/memory/memory.limit_in_bytes
else
    echo "Memory limit file not found"
fi

echo -e "\n=== Docker Run Configuration ==="
echo "Checking devcontainer.json settings..."
if [ -f .devcontainer/devcontainer.json ]; then
    grep -A 5 "shm-size" .devcontainer/devcontainer.json
else
    echo "devcontainer.json not found"
fi 
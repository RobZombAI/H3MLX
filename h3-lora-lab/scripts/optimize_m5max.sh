#!/bin/bash
set -e
echo "=== M5 Max AI Inference Optimization ==="

# 1. High Power Mode (sblocca frequenze massime GPU/CPU)
sudo pmset -a powermode 2
sudo pmset -a disablesleep 1

# 2. GPU Wired Memory → 120 GB (dei 128 GB totali)
sudo sysctl iogpu.wired_limit_mb=122880

# 3. File descriptors
sudo sysctl -w kern.maxfiles=1048576 2>/dev/null || true
ulimit -n 65536

echo "✅ Sistema ottimizzato per inference AI"

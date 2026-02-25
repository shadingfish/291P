#!/usr/bin/env bash
# Run all suggested CLI commands for horizontal comparison.
# From project root 291P:  bash scripts/run_all_commands.sh
# Or:  PYTHONPATH=. python run_cli.py run-all   # single CSV table

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "=== 1. Single AllReduce (M=140GB), vary topology and N ==="
python run_cli.py collective --topology ring --N 8 --M 140e9
python run_cli.py collective --topology tree --N 8 --M 140e9
python run_cli.py collective --topology hierarchical --N 8 --M 140e9
python run_cli.py collective --topology hierarchical --N 8 --gpus-per-node 4 --M 140e9
python run_cli.py collective --topology switch --N 8 --M 140e9
python run_cli.py collective --topology torus --N 8 --M 140e9
python run_cli.py collective --topology torus --N 8 --grid-nx 2 --grid-ny 4 --M 140e9
python run_cli.py collective --topology ring --N 16 --M 140e9
python run_cli.py collective --topology tree --N 16 --M 140e9
python run_cli.py collective --topology hierarchical --N 16 --M 140e9
python run_cli.py collective --topology hierarchical --N 16 --gpus-per-node 8 --M 140e9
python run_cli.py collective --topology hierarchical --N 16 --gpus-per-node 4 --M 140e9
python run_cli.py collective --topology switch --N 16 --M 140e9
python run_cli.py collective --topology torus --N 16 --M 140e9
python run_cli.py collective --topology torus --N 16 --grid-nx 4 --grid-ny 4 --M 140e9


echo ""
echo "=== 2. Full analysis (DP=8, 70B) ==="
python run_cli.py analysis --topology ring --num-gpus 8 --dp 8 --params 70e9
python run_cli.py analysis --topology tree --num-gpus 8 --dp 8 --params 70e9
python run_cli.py analysis --topology hierarchical --num-gpus 8 --dp 8 --params 70e9
python run_cli.py analysis --topology hierarchical --num-gpus 8 --dp 8 --gpus-per-node 4 --params 70e9
python run_cli.py analysis --topology switch --num-gpus 8 --dp 8 --params 70e9
python run_cli.py analysis --topology torus --num-gpus 8 --dp 8 --params 70e9

echo ""
echo "=== 3. Memory only (no P): per-GPU memory for 70B ==="
python run_cli.py analysis --topology ring --num-gpus 1 --dp 1 --tp 1 --cp 1 --params 70e9

echo ""
echo "=== 4. With TP=2 ==="
python run_cli.py analysis --topology ring --num-gpus 8 --dp 4 --tp 2 --params 70e9 --seq-length 1024 --hidden-size 4096
python run_cli.py analysis --topology hierarchical --num-gpus 8 --dp 4 --tp 2 --gpus-per-node 4 --params 70e9 --seq-length 1024 --hidden-size 4096

echo ""
echo "=== 5. With CP=4 ==="
python run_cli.py analysis --topology ring --num-gpus 8 --dp 2 --cp 4 --params 70e9 --seq-length 1024 --hidden-size 4096
python run_cli.py analysis --topology hierarchical --num-gpus 8 --dp 2 --cp 4 --gpus-per-node 4 --params 70e9 --seq-length 1024 --hidden-size 4096
python run_cli.py analysis --topology switch --num-gpus 8 --dp 4 --tp 2 --params 70e9 --seq-length 1024 --hidden-size 4096
python run_cli.py analysis --topology torus --num-gpus 8 --dp 4 --tp 2 --params 70e9 --seq-length 1024 --hidden-size 4096

echo ""
echo "=== 6. DP+TP+CP with num-layers (combined total comm/step) ==="
python run_cli.py analysis --topology ring --num-gpus 8 --dp 2 --tp 2 --cp 2 --params 70e9 --seq-length 1024 --hidden-size 4096 --num-layers 32
python run_cli.py analysis --topology switch --num-gpus 8 --dp 2 --tp 2 --cp 2 --params 70e9 --seq-length 1024 --hidden-size 4096 --num-layers 32
python run_cli.py analysis --topology torus --num-gpus 8 --dp 2 --tp 2 --cp 2 --params 70e9 --seq-length 1024 --hidden-size 4096 --num-layers 32

echo ""
echo "=== 7. Verbose (formula and calculation) ==="
python run_cli.py collective --topology hierarchical --N 8 --gpus-per-node 4 --M 140e9 --verbose
python run_cli.py collective --topology torus --N 8 --grid-nx 2 --grid-ny 4 --M 140e9 --verbose

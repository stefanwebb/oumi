#!/bin/bash

#SBATCH --job-name=multinode-training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=DEBUG
source activate base
conda activate oumi-new
export HF_HOME=/data/cache/hf
export NCCL_DEBUG=TRACE

srun oumi train \
-c configs/recipes/falcon_h1/dpo/falcon_h1_0_5b/qlora_dpo.yaml

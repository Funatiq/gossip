#!/bin/bash

#SBATCH -J gossip_benchmark
#SBATCH -o gossip_benchmark.%j.out
#SBATCH -t 05:00:00
#SBATCH -A pararch
#SBATCH -p infogpu
#SBATCH --gres=gpu:4
#SBATCH -w ps0001

module purge
module load compiler/GCC/6.3.0

cd /home/djuenger/projects/gossip
time python benchmark.py
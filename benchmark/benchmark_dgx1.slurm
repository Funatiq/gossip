#!/bin/bash

#SBATCH -J gossip_benchmark
#SBATCH -o gossip_benchmark.%j.out
#SBATCH -t 03:00:00
#SBATCH -A pararch
#SBATCH -p deeplearning
#SBATCH -c 20
#SBATCH --mem 32g
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -w dgx01

export PATH=$PATH:/bin

cd /home/djuenger/project/djuenger/gossip/benchmark

#time python benchmark.py dgx1_symm
#time python benchmark.py dgx1_opt
#time python3 benchmark.py dgx1_opt_chunk
#time python benchmark.py dgx1_direct
#time python benchmark.py dgx1_rings

time python3 benchmark.py dgx1_quad_opt2

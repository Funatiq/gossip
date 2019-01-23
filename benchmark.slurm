#!/bin/bash

#SBATCH -J gossip_benchmark
#SBATCH -o gossip_benchmark.%j.out
#SBATCH -t 05:00:00
#SBATCH -A pararch
#SBATCH -p deeplearning
#SBATCH -c 40
#SBATCH --mem 400000
#SBATCH --gres=gpu:8
#SBATCH -w dgx01

export PATH=$PATH:/bin

module purge
module load compiler/GCC/6.3.0

cd /home/djuenger/projects/gossip

#time python benchmark.py dgx1_symm
#time python benchmark.py dgx1_opt
time python benchmark.py dgx1_opt_1chunk
#time python benchmark.py dgx1_direct
#time python benchmark.py dgx1_rings
#time python benchmark.py dgx1_symm
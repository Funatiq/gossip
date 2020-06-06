#!/usr/bin/env python3

from subprocess import run, PIPE
import numpy as np

def get_topology_matrix(filename = ""):
    if filename:
        with open(filename, "r") as file:
            lines = file.read().split('\n')
    else:
        process = run(["nvidia-smi", "topo", "-m"], stdout=PIPE, universal_newlines=True)
        lines = process.stdout.split('\n')

    topology_lines = [l for l in lines if l.find("GPU") == 0]

    num_gpus = len(topology_lines)

    nvlink = False
    topology = []
    for line in topology_lines:
        if line.find("NV") >= 0:
            nvlink = True
        topology.append(line.split()[1:num_gpus+1])

    if not nvlink:
        return np.ones((num_gpus,num_gpus))

    topology_matrix = np.eye(num_gpus) * num_gpus

    for i in range(len(topology)):
        for j in range(len(topology[i])):
            item = topology[i][j]
            if item[:2] == "NV":
                topology_matrix[i,j] = int(item[2:])

    return topology_matrix

# topology_matrix = get_topology_matrix()
# print(topology_matrix)

# topology_matrix = get_topology_matrix("dgx1_topology.txt")
# print(topology_matrix)
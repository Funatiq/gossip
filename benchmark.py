import sys
import subprocess
import os

all2all_csv = ""
scatter_csv = ""
gather_csv = ""

def valid(output):
    for line in output.split('\n'):
        if line.startswith('ERROR'):
            return False
    return True

sizes = [2**exp for exp in range(12, 30)]
repeats = 10
file_suffix = 'benchmark/ps0001_general'
exe = "./general"

for i, s in enumerate(sizes):
    print("PROGRESS: " + str(i+1) + "/" + str(len(sizes)))

    out = subprocess.check_output([exe, str(s)])

    for line in out.split('\n'):
        if line.startswith('INFO') and line.endswith('(all2all)'):
            all2all_csv += str(line.split(' ')[1])
        if line.startswith('INFO') and line.endswith('(scatter/gather)'):
            scatter_csv += str(line.split(' ')[1])
            gather_csv += str(line.split(' ')[1])

    for r in range(repeats):
        out = subprocess.check_output([exe, str(s)])
        
        if(not valid(out)):
            print("ERROR at size=" + str(s) + " repeat=" + str(r+1))
            print(out)
            continue

        for line in out.split('\n'):
            if line.startswith('TIMING') and line.endswith('(all2all)'):
                all2all_csv += "," + str(float(line.split(' ')[1]))
            if line.startswith('TIMING') and line.endswith('(scatter)'):
                scatter_csv += "," + str(float(line.split(' ')[1]))
            if line.startswith('TIMING') and line.endswith('(gather)'):
                gather_csv += "," + str(float(line.split(' ')[1]))

    if(i < len(sizes)-1):
        all2all_csv += '\n'
        scatter_csv += '\n'
        gather_csv += '\n'

with open("all2all_" + file_suffix + ".csv", "w+") as f:
        f.write(all2all_csv)
with open("scatter_" + file_suffix + ".csv", "w+") as f:
        f.write(scatter_csv)
with open("gather_" + file_suffix + ".csv", "w+") as f:
        f.write(gather_csv)

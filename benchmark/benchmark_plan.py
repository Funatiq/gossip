import sys
import subprocess
import argparse
import os
from pathlib import Path

def absolute_path(path):
    return str(Path(path).resolve())

# check if output shows no errors
def valid(output):
    for line in output.decode().split('\n'):
        if 'error' in line.lower():
            return False
    return True

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("type", type=str, help="collective type (all2all|all2all_async|scatter_gather|broadcast)")
parser.add_argument("plan", type=str, nargs='+', help="JSON which specifies the communication strategy\n\
    scatter_gather requires one plan for each collective")
parser.add_argument("--output", "-o", type=str, help="output file", default="benchmark.csv")
parser.add_argument("--repeats", "-r", type=int, help="number of repeated executions", default=3)
parser.add_argument("--maxsize", type=int, help="maximum overall amount of data to be communicated (bytes log2)", default=28)
parser.add_argument("--minsize", type=int, help="minimum overall amount of data to be communicated (bytes log2)", default=12)
args = parser.parse_args()

# args.type has to be one of the following options
types = ["all2all", "all2all_async", "scatter_gather", "broadcast"]
assert(args.type in types)
assert(args.output.endswith(".csv"))
if args.type == "scatter_gather":
    assert(len(args.plan) >= 2)

base_path = str(Path(__file__).parent.parent.resolve())

# execute collective for a range of data sizes
sizes = range(args.minsize, args.maxsize) # max: dgxv1=28, dgx1v2=29, dgx2=30

# calls the collective
exe = base_path + "/execute"
# output is a csv file, where each line has the form <data size in bytes>,<runtime in ms>,<runtime in ms>,...
out_csv = ""
# extract filename to use as output file descriptor
plans = [absolute_path(plan) for plan in args.plan]

# main loop over data sizes
for i, s in enumerate(sizes):
    print("\tPROGRESS: size " + str(i+1) + "/" + str(len(sizes)))

    # secondary loop over repeats
    for r in range(args.repeats):
        print("\t\tPROGRESS: repeat " + str(r+1) + "/" + str(args.repeats))

        # execute collective
        if args.type == "scatter_gather":
            out = subprocess.check_output([exe, args.type, plans[0], plans[1], "--size", str(s)])
        else:
            out = subprocess.check_output([exe, args.type, plans[0], "--size", str(s)])

        # process result and extract runtime
        for line in out.decode().split('\n'):
            if r == 0:
                # add data size [bytes] as first columns to csv
                if line.startswith('INFO') and line.endswith("(" + args.type + ")"):
                    out_csv += str(line.split(' ')[1])

            # add runtime [ms] as subsequent column(s) to csv
            if args.type == "scatter_gather":
                # only measure scatter since both operations perform the same
                if line.startswith('TIMING') and line.endswith("(" + "scatter" + ")"):
                    out_csv += "," + str(float(line.split(' ')[1]))
            else:
                if line.startswith('TIMING') and line.endswith("(" + args.type + ")"):
                    out_csv += "," + str(float(line.split(' ')[1]))

        # check if output shows any errors
        if(not valid(out)):
            print("ERROR at size=" + str(s) + " repeat=" + str(r+1))
            print(out)
            sys.exit(1)

    if(i < len(sizes)-1):
        out_csv += '\n'

# write output to csv
with open(args.output, "w+") as f:
        f.write(out_csv)

print("done!")

import sys
import subprocess
import argparse
import os
import signal
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("arch", type=str, help="architecture (dgx1|dgx2)")
parser.add_argument("--dir", type=str, help="output directory", default=".")
parser.add_argument("--python", type=str, help="python interpreter", default="python3")
args = parser.parse_args()

if args.dir.endswith('/'):
    args.dir = args.dir[:-1]

# args.arch has to be one of the following options
types = ["dgx1", "dgx2"]
assert(args.arch in types)

base_path = str(Path(__file__).parent.parent.resolve())

# relevant plans for DGX-1 topology
dgx1_plans = [
    base_path + "/plans/dgx1_direct/all2all_plan.json",
    base_path + "/plans/dgx1_opt/all2all_plan.json",
    base_path + "/plans/dgx1_rings/all2all_plan.json"
]

# relevant plans for DGX-2 topology
dgx2_plans = [
    base_path + "/plans/dgx2_direct/all2all_plan.json",
    base_path + "/plans/dgx2_opt/all2all_plan.json"
]

plans = {"dgx1": dgx1_plans, "dgx2": dgx2_plans}

# mkdir
if not os.path.exists(args.dir):
    os.makedirs(args.dir)

# make sure subprocesses return at SIGINT
def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

for i, plan in enumerate(plans[args.arch]):
    print("PROGRESS: plan " + str(i+1) + "/" + str(len(plans[args.arch])))

    plan_label = plan.split('/')[-2]

    print("\tall2all")
    p1 = subprocess.Popen([args.python, base_path + "/benchmark/benchmark_plan.py", "all2all", plan, "-o", args.dir + "/" + plan_label + "_all2all.csv"]).wait()
    print("\tall2all_async")
    p2 = subprocess.Popen([args.python, base_path + "/benchmark/benchmark_plan.py", "all2all_async", plan, "-o", args.dir + "/" + plan_label + "_all2all_async.csv"]).wait()

    if p1 or p2:
        print("ERROR: subprocess terminated with non-zero exit code")
        sys.exit(1)


subprocess.Popen([args.python, base_path + "/benchmark/plot_results.py", "-o", args.dir + "/" + args.arch + "_all2all_benchmark.pdf", args.dir + "/*"]).wait()
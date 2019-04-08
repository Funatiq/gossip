import sys
import numpy as np
import csv
import glob
import argparse
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-paper')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("csvs", nargs='+', type=str, help="csv traces to be included in the plot e.g. <path%%label%%color> or <path%%label> or <path>",)
parser.add_argument("--output", "-o", type=str, help="output file", default="show plot")
parser.add_argument("--reduction", "-r", type=str, help="reduction operation of multiple runs (median|mean|min|max)", default="median")
args = parser.parse_args()

# args.reduction has to be one of the following options
reductions = {"median": np.median, "mean": np.mean, "min": np.min, "max": np.max}
assert(args.reduction in reductions)

def handle_args(args):
    ret_filenames = []
    ret_labels = []
    ret_colors = []

    for a in args:
        splits = a.split('%')
        filenames = [f for f in glob.glob(splits[0]) if f.endswith(".csv")]
        labels = [f.split('/')[-1].split(".")[:-1][0] for f in filenames]
        colors = [None] * len(filenames)
    
        if len(filenames) is 1:
            if(len(splits) > 1):
                labels = [splits[1]]

            if(len(splits) > 2):
                colors = [splits[2]]

        ret_filenames.extend(filenames)
        ret_labels.extend(labels)
        ret_colors.extend(colors)

    return zip(ret_filenames, ret_labels, ret_colors)

# convert csv to trace by reducing multiple runs to a scalars
def csv_to_trace(filename, reduce=np.median):
    xs, ys = [], []
    with open(filename, 'r') as file:
        csv_file = csv.reader(file, delimiter=',')

        for row in csv_file:
            xs.append(int(row[0]))
            y = reduce([float(x) for x in row[1:]])
            ys.append(y)
    return xs, ys

def bandwidth(bs, ts):
    return [((float(b))/1024**3)/(t/1000.0) for b, t in zip(bs, ts)]

fig, ax = plt.subplots()
ax.set_xscale("log", basex=2, nonposx='clip')
#ax.set_yscale("log", basey=10, nonposy='clip')
ax.grid()

config = handle_args(args.csvs)

for c in config:
    xs, ys = csv_to_trace(c[0], reductions[args.reduction])
    bw = bandwidth(xs, ys)
    print("trace: " + str(c) + ", peak troughput: " + str(int(np.max(bw))) + " GB/s")

    ax.scatter(xs, bw, c=c[2])
    ax.plot(xs, bw, label=c[1], c=c[2])

plt.xlabel("Input Size [Bytes]")
plt.ylabel("Throughput [GB/s]")

plt.legend()

# either show plot or save as file
if args.output == "show plot":
    plt.show()
    print("plot displayed")
else:
    plt.savefig(args.output, transparent=True, bbox_inches='tight')
    print("plot saved as " + args.output)

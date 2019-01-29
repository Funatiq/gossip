import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
plt.style.use('ggplot')

filenames = sys.argv[1:]

def csv_to_trace(filename, reduce=np.amin):
    xs, ys = [], []
    with open(filename, 'r') as file:
        csv_file = csv.reader(file, delimiter=',')
        for row in csv_file:
            xs.append(int(row[0]))
            y = reduce([float(x) for x in row[1:]])
            ys.append(y)
    return xs[1:], ys[1:]

def bandwidth(bs, ts):
    return [(((0.875)*float(b))/1024**3)/(t/1000.0) for b, t in zip(bs, ts)]
#    return [(((0.5)*float(b))/1024**3)/(t/1000.0) for b, t in zip(bs, ts)]

fig, ax = plt.subplots()
ax.set_xscale("log", basex=2, nonposx='clip')
#ax.set_yscale("log", basey=10, nonposy='clip')
xs, ys = [], []

for filename in filenames:
    xs, ys = csv_to_trace(filename, np.median)
    print(filename)
    label = filename.split('.')[0]
    ax.plot(xs, bandwidth(xs, ys), label=label)
    ax.scatter(xs, bandwidth(xs, ys), marker='+')

plt.xlabel("transfer size [bytes]")
plt.ylabel("bandwidth [GB/s]")
plt.legend()
plt.savefig("plot.pdf")

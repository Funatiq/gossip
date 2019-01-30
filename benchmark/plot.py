import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
#plt.style.use('ggplot')

configs = [{"V1": "dgx1/all2all_async/direct.csv",
            "V2": "dgx1/all2all_async/rings.csv",
            "V3": "dgx1/all2all_async/opt_1chunk.csv",
            "V4": "dgx1/all2all_async/opt.csv"},
           {"V1": "dgx1/scatter/direct.csv",
            "V2": "dgx1/scatter/rings.csv",
            "V3": "dgx1/scatter/opt_1chunk.csv",
            "V4": "dgx1/scatter/opt.csv"},
           {"V1": "dgx1/gather/direct.csv",
            "V2": "dgx1/gather/rings.csv",
            "V3": "dgx1/gather/opt_1chunk.csv",
            "V4": "dgx1/gather/opt.csv"},
           {"V1": "p100_quad/all2all_async/direct.csv",
            "V2": "p100_quad/all2all_async/rings.csv",
            "V3": "p100_quad/all2all_async/opt.csv"},
           {"V1": "p100_quad/scatter/direct.csv",
            "V2": "p100_quad/scatter/rings.csv",
            "V3": "p100_quad/scatter/opt.csv"},
           {"V1": "p100_quad/gather/direct.csv",
            "V2": "p100_quad/gather/rings.csv",
            "V3": "p100_quad/gather/opt.csv"},
           {"V1": "dgx1_quad/all2all_async/direct_p100.csv",
            "V2": "dgx1_quad/all2all_async/rings_p100.csv",
            "V3": "dgx1_quad/all2all_async/opt_p100.csv"},
           {"V1": "dgx1_quad/scatter/direct_p100.csv",
            "V2": "dgx1_quad/scatter/rings_p100.csv",
            "V3 (a)": "dgx1_quad/scatter/opt_p100.csv",
            "V3 (b)": "dgx1_quad/scatter/opt.csv"},
           {"V1": "dgx1_quad/gather/direct_p100.csv",
            "V2": "dgx1_quad/gather/rings_p100.csv",
            "V3 (a)": "dgx1_quad/gather/opt_p100.csv",
            "V3 (b)": "dgx1_quad/gather/opt.csv"}]

filenames = {}
if len(sys.argv) < 2:
    filenames = configs[3]
else:
    for filename in sys.argv[1:]:
        filenames[filename.split('.')[0]] = filename

def csv_to_trace(filename, reduce=np.median):
    xs, ys = [], []
    with open(filename, 'r') as file:
        csv_file = csv.reader(file, delimiter=',')
        for row in csv_file:
            xs.append(int(row[0]))
            y = reduce([float(x) for x in row[1:]])
            ys.append(y)
    return xs[1:], ys[1:]

def bandwidth(bs, ts):
    return [((float(b))/1024**3)/(t/1000.0) for b, t in zip(bs, ts)]
#    return [(((0.5)*float(b))/1024**3)/(t/1000.0) for b, t in zip(bs, ts)]

fig, ax = plt.subplots()
ax.set_xscale("log", basex=2, nonposx='clip')
#ax.set_yscale("log", basey=10, nonposy='clip')
xs, ys = [], []
ax.grid()
for label, filename in filenames.items():
    xs, ys = csv_to_trace(filename, np.median)
    print(filename)
    ax.plot(xs, bandwidth(xs, ys), label=label)
    ax.scatter(xs, bandwidth(xs, ys), marker='+')

plt.xlabel("input size [bytes]")
plt.ylabel("throughput [GB/s]")

plt.legend()
plt.show()
#plt.savefig("plot.pdf")

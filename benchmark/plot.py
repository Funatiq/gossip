import sys
import numpy as np
import csv
import matplotlib.pyplot as plt

plt.style.use('seaborn-paper')
#plt.rc('text', usetex=True)

default_marker = '.'
default_linestyle = '-'
params = {'V1':   {'color': 'black', 
                   'marker': default_marker, 
                   'linestyle': default_linestyle, 
                   'label': 'V1 Direct'},
          'V2':   {'color': 'green', 
                   'marker': default_marker, 
                   'linestyle': default_linestyle, 
                   'label': 'V2 Rings'},
          'V3':   {'color': 'royalblue', 
                   'marker': default_marker, 
                   'linestyle': default_linestyle, 
                   'label': 'V3 Opt. unsplit'},
          'V4':   {'color': 'firebrick', 
                   'marker': default_marker, 
                   'linestyle': default_linestyle, 
                   'label': "V4 Opt. k-split"},
          'V4_1': {'color': 'firebrick', 
                   'marker': default_marker, 
                   'linestyle': default_linestyle, 
                   'label': "V4 Opt. 1-split"},
          'V4_1--': {'color': 'firebrick', 
                   'marker': default_marker, 
                   'linestyle': '--', 
                   'label': "V4 Opt. 1-split"},
          'V4_3': {'color': 'firebrick', 
                   'marker': default_marker, 
                   'linestyle': default_linestyle, 
                   'label': "V4 Opt. 3-split"},
          'V4_4': {'color': 'firebrick', 
                   'marker': default_marker, 
                   'linestyle': default_linestyle, 
                   'label': "V4 Opt. 4-split"},
          'V4_4--': {'color': 'firebrick', 
                   'marker': default_marker, 
                   'linestyle': '--', 
                   'label': "V4 Opt. 4-split"},
          'V4_5': {'color': 'firebrick', 
                   'marker': default_marker, 
                   'linestyle': default_linestyle, 
                   'label': "V4 Opt. 5-split"},
          'V4_6': {'color': 'firebrick', 
                   'marker': default_marker, 
                   'linestyle': default_linestyle, 
                   'label': "V4 Opt. 6-split"},
          'V4_k': {'color': 'firebrick', 
                   'marker': default_marker, 
                   'linestyle': default_linestyle, 
                   'label': "V4 Opt. k-split"},
          'V5':   {'color': 'goldenrod', 
                   'marker': default_marker, 
                   'linestyle': default_linestyle, 
                   'label': "V5 Double-buffered"}}

csvs = {"dgx1_all2all":      {"V1": "results/dgx1/all2all_async/direct.csv",
                              "V2": "results/dgx1/all2all_async/rings.csv",
                              "V3": "results/dgx1/all2all_async/opt_1chunk.csv",
                              "V4_3": "results/dgx1/all2all_async/opt.csv",
                              "V5": "results/dgx1/all2all/opt.csv"},
        "dgx1_scatter":      {"V1": "results/dgx1/scatter/direct.csv",
                              "V2": "results/dgx1/scatter/rings.csv",
                              "V3": "results/dgx1/scatter/opt_1chunk.csv",
                              "V4_6": "results/dgx1/scatter/opt.csv"},
        "dgx1_gather":       {"V1": "results/dgx1/gather/direct.csv",
                              "V2": "results/dgx1/gather/rings.csv",
                              "V3": "results/dgx1/gather/opt_1chunk.csv",
                              "V4_6": "results/dgx1/gather/opt.csv"},
        "p100_quad_all2all": {"V1": "results/p100_quad/all2all_async/direct.csv",
                              "V2": "results/p100_quad/all2all_async/rings.csv",
                              "V4_1": "results/p100_quad/all2all_async/opt.csv"},
        "p100_quad_scatter": {"V1": "results/p100_quad/scatter/direct.csv",
                              "V2": "results/p100_quad/scatter/rings.csv",
                              "V4_4": "results/p100_quad/scatter/opt.csv"},
        "p100_quad_gather":  {"V1": "results/p100_quad/gather/direct.csv",
                              "V2": "results/p100_quad/gather/rings.csv",
                              "V4_4": "results/p100_quad/gather/opt.csv"},
        "dgx1_quad_all2all": {"V1": "results/dgx1_quad/all2all_async/direct_p100.csv",
                              "V2": "results/dgx1_quad/all2all_async/rings_p100.csv",
                              "V4_1--": "results/dgx1_quad/all2all_async/opt_p100.csv",
                              "V4_5": "results/dgx1_quad/all2all_async/opt.csv"},
        "dgx1_quad_scatter": {"V1": "results/dgx1_quad/scatter/direct_p100.csv",
                              "V2": "results/dgx1_quad/scatter/rings_p100.csv",
                              "V4_4--": "results/dgx1_quad/scatter/opt_p100.csv",
                              "V4_5": "results/dgx1_quad/scatter/opt.csv"},
        "dgx1_quad_gather":  {"V1": "results/dgx1_quad/gather/direct_p100.csv",
                              "V2": "results/dgx1_quad/gather/rings_p100.csv",
                              "V4_4--": "results/dgx1_quad/gather/opt_p100.csv",
                              "V4_5": "results/dgx1_quad/gather/opt.csv"}}

setups = []
plot_all = len(sys.argv) < 2
if plot_all:
    setups = ["dgx1_all2all", "dgx1_scatter",
              "p100_quad_all2all", "p100_quad_scatter",
              "dgx1_quad_all2all", "dgx1_quad_scatter"]
else:
    setups = sys.argv[1:]

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

for setup in setups:
    print(setup)

    fig, ax = plt.subplots()

    #plt.tight_layout(w_pad=1.6, h_pad=1.0)
    ax.set_xscale("log", basex=2, nonposx='clip')
    #ax.set_yscale("log", basey=10, nonposy='clip')

    xs, ys = [], []
    ax.grid()
    
    for key, filename in csvs[setup].items():
        xs, ys = csv_to_trace(filename, np.median)
        bw = bandwidth(xs, ys)
        print('\t' + filename + " peak troughput: " + str(int(np.max(bw))) + " GB/s")
        #print("\tThroughput " + str(bw))

        p = params[key]
        ax.scatter(xs, bw, color=p['color'], marker=p['marker'])
        ax.plot(xs, bw, **p)
        

        plt.xlabel("Input Size [Bytes]")
        plt.ylabel("Throughput [GB/s]")

        plt.legend()
        #plt.show()
        plt.savefig("plots/" + setup + ".pdf", transparent=True, bbox_inches='tight')

#!/usr/bin/env python3

from itertools import cycle, islice, dropwhile
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser(description="create transfer plan.")
parser.add_argument("mode", type=str, help="scatter, gather or all2all")
parser.add_argument("-m","--main_gpu", type=int, default=0, help="source for scatter or target for gather")
args=parser.parse_args()

modes = ["scatter", "gather", "all2all", "broadcast"]
if args.mode in modes:
    mode = args.mode
else:
    print("invalid mode")
    parser.print_help()
    raise SystemExit

main_gpu = args.main_gpu


# ps0001 4x pascal: 4 nvlink per gpu, 2 rings
num_gpus = 4
rings = [[0,1,2,3],
         [0,2,1,3]]
# hypotetical 4x volta: 6 nvlink per gpu, 3 rings
# num_gpus = 4
# rings = [[0,1,2,3],
#          [0,1,3,2],
#          [0,2,1,3]]
# dgx1 8x volta: 6 nvlink per gpu, 3 rings
# num_gpus = 8
# rings = [[0,4,7,6,5,1,2,3],
#          [0,4,7,6,5,1,2,3],
#          [0,1,3,7,5,4,6,2]]

half_num_gpus = num_gpus//2
num_chunks = 2*len(rings)


def make_paths(ring, src, forward=True, wait=True):
    plan = []
    chunks = []

    for i in range(half_num_gpus):
        length = half_num_gpus-i
        if forward:
            cycled = cycle(ring)
        else:
            cycled = cycle(reversed(ring))
        skipped = dropwhile(lambda x: x != src, cycled)
        path = islice(skipped, 0, length+1)
        path = list(path)
        if wait:
            wait_steps = (half_num_gpus*(half_num_gpus+1)//2) - ((half_num_gpus-i)*(half_num_gpus-i+1)//2)
            fill_steps = ((half_num_gpus-i-1)*(half_num_gpus-i)//2)
        else:
            wait_steps = i
            fill_steps = 0
        full_path = [path[0]]*wait_steps + path + [path[-1]]*fill_steps
        # print(path)
        # print(full_path)
        plan.append(full_path)
        if i == 0 and num_gpus%2 == 0:
            chunk = 1
        else:
            chunk = 2
        chunks.append(chunk)

    return plan, chunks

def make_all2all_plan():
    plan = []
    chunks = []

    # copy to self
    for src in range(num_gpus):
        steps = ((half_num_gpus)*(half_num_gpus+1)//2)+1
        path = [src] * steps
        plan.append(path)
        chunks.append(num_chunks)
    # transfer along rings
    for ring in rings:
        for src in range(num_gpus):
            # foward
            i_plan, i_chunks = make_paths(ring, src, forward=True)
            plan += i_plan
            chunks += i_chunks
            # reverse
            i_plan, i_chunks = make_paths(ring, src, forward=False)
            plan += i_plan
            chunks += i_chunks

    chunk_counter = np.zeros((num_gpus,num_gpus))
    for p,c in zip(plan,chunks):
        # if p[0] == 0: print(p,c)
        chunk_counter[p[0],p[-1]] += c
    assert(np.all(chunk_counter == num_chunks))

    return plan, chunks

def make_scatter_plan(src):
    plan = []
    chunks = []

    # copy to self
    steps = half_num_gpus+1
    path = [src] * steps
    plan.append(path)
    chunks.append(num_chunks)
    # transfer along rings
    for ring in rings:
        # foward
        i_plan, i_chunks = make_paths(ring, src, forward=True, wait=False)
        plan += i_plan
        chunks += i_chunks
        # reverse
        i_plan, i_chunks = make_paths(ring, src, forward=False, wait=False)
        plan += i_plan
        chunks += i_chunks

    chunk_counter = np.zeros(num_gpus)
    for p,c in zip(plan,chunks):
        # print(p,c)
        # assert(plan[0] == src)
        chunk_counter[p[-1]] += c
    # print(chunk_counter)
    assert(np.all(chunk_counter == num_chunks))

    return plan, chunks

def make_gather_plan(trg):
    plan,chunks = make_scatter_plan(src=trg)
    plan = [list(reversed(p)) for p in plan]
    return plan, chunks

def make_broadcast_plan(src, chunks_per_ring):
    plan = []
    chunks = []

    length = len(rings[0])
    path_length = len(rings[0]) + chunks_per_ring - 1 + (len(rings)-1) * chunks_per_ring

    for r,ring in enumerate(rings):
        # forward
        for c in range(chunks_per_ring):
            # to self
            plan.append([src]* path_length)
            chunks.append(r*chunks_per_ring*2+c)
            # to others
            cycled = cycle(ring)
            skipped = dropwhile(lambda x: x != src, cycled)
            path = islice(skipped, 0, length)
            path = list(path)
            path = [path[0]] * c + path + [path[-1]] * (chunks_per_ring - c - 1)
            path = [path[0]] * r * chunks_per_ring + path + [path[-1]] * (len(rings)-r-1) * chunks_per_ring
            plan.append(path)
            chunks.append(r*chunks_per_ring*2+c)
        # reverse
        for c in range(chunks_per_ring):
            # to self
            plan.append([src]* path_length)
            chunks.append(r*chunks_per_ring*2+chunks_per_ring+c)
            # to others
            cycled = cycle(reversed(ring))
            skipped = dropwhile(lambda x: x != src, cycled)
            path = islice(skipped, 0, length)
            path = list(path)
            path = [path[0]] * c + path + [path[-1]] * (chunks_per_ring - c - 1)
            path = [path[0]] * r * chunks_per_ring + path + [path[-1]] * (len(rings)-r-1) * chunks_per_ring
            plan.append(path)
            chunks.append(r*chunks_per_ring*2+chunks_per_ring+c)
    return plan, chunks

if mode == "all2all":
    plan,chunks = make_all2all_plan()
    for p,c in zip(plan,chunks):
        if p[0] == main_gpu: print(p,c)
elif mode == "scatter":
    plan,chunks = make_scatter_plan(src=main_gpu)
    for p,c in zip(plan,chunks):
        print(p,c)
elif mode == "gather":
    plan,chunks = make_gather_plan(trg=main_gpu)
    for p,c in zip(plan,chunks):
        print(p,c)
elif mode == "broadcast":
    plan,chunks = make_broadcast_plan(src=main_gpu, chunks_per_ring=20)
    num_chunks = len(plan) // 2
    for p,c in zip(plan,chunks):
        print(p,c)
else:
    raise SystemExit

steps = len(plan[0]) - 1

data = {
    "type" : mode,
    "num_gpus" : num_gpus,
    "main_gpu" : main_gpu,
    "num_steps" : steps,
    "num_chunks" : num_chunks,
    "plan" : plan,
    "chunks" : chunks
}
# print(data)

json_string = json.dumps(data)
print(json_string)
json_name = mode+"_plan.json"
print("saving json to '%s'" %(json_name))
with open(json_name, "w") as file:
    json.dump(data, file)

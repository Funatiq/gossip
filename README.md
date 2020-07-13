# `gossip`: Efficient Communication Primitives for Multi-GPU Systems

Gossip supports scatter, gather and all-to-all communication. To execute one of the communication primitives a transfer plan is needed. Use the provided [scripts](scripts) to generate optimized plans for your specific NVLink topology. The [plans directory](plans) contains optimized plans for typical 4 GPU configurations ([P100](plans/p100_quad_opt/) and [V100](plans/v100_quad_opt/)) as well as 8 GPU [DGX-1 Volta](plans/dgx1_opt). If no transfer plan is provided gossip will fall back to the default strategy using direct transfers between GPUs.

Gossip was presented at [ICPP '19](https://dl.acm.org/citation.cfm?id=3337889).


## Using gossip

To use gossip clone this repository and check out the submodule *cudahelpers* by calling `submodule update --init include/cudahelpers`. Include the header [gossip.cuh](include/gossip.cuh) in your project which provides all communication primitives. To parse transfer plans make use of the [plan parser](include/plan_parser.hpp) which can be compiled as a separate unit like in the example [Makefile](Makefile).


## Examples

The example [execute.cu](execute.cu) executes gossip's communication primitives on uniformly distributed random numbers. The data is first split into a number of chunks corresponding to the number of GPUs (multisplit). The chunks sizes are displayed as a partiton table (row=source GPU, column=target GPU). Then the data is transferred between the GPUs according to the provided transfer plan. At the end it validates if all data reached the correct destinations.

The example [simulate.cu](simulate.cu) allows to run the multi-GPU example above simulated on a single GPU.

### Build example

Compile the example using the provided [Makefile](Makefile) by calling `git submodule update --init && make`.

Requirements:

- CUDA >= 9.2
- GNU g++ >= 5.5 compatible with your CUDA version
- Python >= 3.0 including
  - Matplotlib
  - NumPy

### Run example

```bash
./execute (all2all|all2all_async) <transfer plan> [--size <size>] [--memory-factor <factor>]

./execute scatter_gather <scatter plan> <gather plan> [--size <size>] [--memory-factor <factor>]
```

Use `./simulate` instead of `./execute` if you want to simulate the example on a single GPU.

Mandatory:

- Choose all2all (double buffered), all2all_async or scatter_gather mode
- Provide path(s) to transfer plan(s) (one for all2all, two for scatter+gather)

Optional:

- Choose data size (2^\<size\> 64-bit elements per GPU) (default: 28)
- Choose memory factor (account for random transfer sizes) (default: 1.5)


## Benchmark

For benchmark scripts and results see the [benchmark directory](benchmark).

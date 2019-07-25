# `gossip`: Efficient Communication Primitives for Multi-GPU Systems

Gossip supports scatter, gather and all-to-all communication. To execute one of the communication primitives a transfer plan is needed. Use the provided [scripts](scripts) to generate optimized plans for your specific NVLink topology. The [plans directory](plans) contains optimized plans for typical 4 GPU configurations ([P100](gossip/plans/p100_quad_opt/) and [V100](gossip/plans/v100_quad_opt/)) as well as 8 GPU [DGX-1 Volta](gossip/plans/dgx1_opt).

Gossip will be presented at ICPP '19. A link to the paper will follow.

## Example

The example [execute.cu](execute.cu) executes gossip's communication primitives on uniformly distributed random numbers. The data is first split into a number of chunks corresponding to the number of GPUs (multisplit). The chunks sizes are displayed as a partiton table (row=source GPU, column=target GPU). Then the data is transferred between the GPUs. At the end it validates if all data reached the correct destinations.

### Build example

Compile the example using the provided `Makefile` by calling `git submodule update --init && make`.

Requirements:

- CUDA >= 9.2
- GNU g++ >= 6.3.0 compatible with your CUDA version
- Python >= 3.0 including
  - Matplotlib
  - NumPy

### Run example

```bash
./execute (all2all|all2all_async) <transfer plan> [--size <size>] [--memory-factor <factor>]

./execute scatter_gather <scatter plan> <gather plan> [--size <size>] [--memory-factor <factor>]
```

Mandatory:

- Choose all2all (double buffered), all2all_async or scatter_gather mode
- Provide path(s) to transfer plan(s) (one for all2all, two for scatter+gather)

Optional:

- Choose data size (2^\<size\> 64-bit elements per GPU) (default: 28)
- Choose memory factor (account for random transfer sizes) (default: 1.5)

## Benchmark

For benchmark scripts and results see the [benchmark directory](benchmark).

NVCC=nvcc
# NVCCFLAGS=-O3 -std=c++14 -arch=sm_70 --expt-extended-lambda -Xcompiler="-fopenmp"
NVCCFLAGS=-O3 -std=c++14 -arch=sm_61 --expt-extended-lambda -Xcompiler="-fopenmp"

HEADERS = include/gossip.cuh \
		  include/gossip/auxiliary.cuh \
		  include/gossip/context.cuh \
		  include/gossip/memory_manager.cuh \
		  include/gossip/multisplit.cuh \
		  include/gossip/point_to_point.cuh

DGX = include/gossip/all_to_all_dgx1v.cuh
GEN = include/gossip/all_to_all.cuh

.PHONY: all clean

all: general dgx1v

general: distributed_general.cu distributed.cuh $(HEADERS) $(GEN)
	$(NVCC) $(NVCCFLAGS) distributed_general.cu -o general

dgx1v: distributed_dgx1v.cu distributed.cuh $(HEADERS) $(DGX)
	$(NVCC) $(NVCCFLAGS) distributed_dgx1v.cu -o dgx1v

clean:
	rm -rf general dgx1v

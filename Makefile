NVCC=nvcc
NVCCFLAGS=-O3 -std=c++14 -arch=sm_70 --expt-extended-lambda -Xcompiler="-fopenmp"

HEADERS = include/gossip.cuh \
		  include/gossip/auxiliary.cuh \
		  include/gossip/context.cuh \
		  include/gossip/memory_manager.cuh \
		  include/gossip/multisplit.cuh \
		  include/gossip/point_to_point.cuh \
		  include/gossip/transfer_plan.hpp

GEN = include/gossip/all_to_all.cuh
DGX = $(GEN) \
      include/gossip/all_to_all_dgx1v.cuh

.PHONY: all clean

all: general dgx1v

general: distributed_general.cu distributed.cuh $(HEADERS) $(GEN) include/plan_parser.cpp include/plan_parser.hpp
	$(NVCC) $(NVCCFLAGS) include/plan_parser.cpp distributed_general.cu -o general

dgx1v: distributed_dgx1v.cu distributed.cuh $(HEADERS) $(DGX)
	$(NVCC) $(NVCCFLAGS) distributed_dgx1v.cu -o dgx1v

clean:
	rm -rf general dgx1v

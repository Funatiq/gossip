NVCC=nvcc
NVCCGENCODE = -gencode arch=compute_60,code=sm_60 \
              -gencode arch=compute_70,code=sm_70
			
NVCCFLAGS = $(NVCCGENCODE) -O3 -std=c++14 --expt-extended-lambda -Xcompiler="-fopenmp" -Wreorder -lineinfo

HEADERS = include/gossip.cuh \
		  include/gossip/all_to_all_async.cuh \
		  include/gossip/all_to_all_plan.hpp \
		  include/gossip/all_to_all.cuh \
		  include/gossip/broadcast_plan.hpp \
		  include/gossip/broadcast.cuh \
		  include/gossip/common.cuh \
		  include/gossip/context.cuh \
		  include/gossip/error_checking.hpp \
		  include/gossip/gather_plan.hpp \
		  include/gossip/gather.cuh \
		  include/gossip/memory_manager.cuh \
		  include/gossip/multisplit.cuh \
		  include/gossip/point_to_point.cuh \
		  include/gossip/scatter_plan.hpp \
		  include/gossip/scatter.cuh \
		  include/gossip/transfer_plan.hpp

.PHONY: all clean

all: execute

execute: execute.cu executor.cuh $(HEADERS) include/plan_parser.cpp include/plan_parser.hpp
	$(NVCC) $(NVCCFLAGS) include/plan_parser.cpp execute.cu -o execute

clean:
	rm -rf execute

NVCC=nvcc
NVCCGENCODE = -gencode arch=compute_60,code=sm_60 \
              -gencode arch=compute_70,code=sm_70
NVCCFLAGS = $(NVCCGENCODE) -O3 -std=c++14 --expt-extended-lambda -Xcompiler="-fopenmp" -Wreorder -lineinfo

HEADERS = include/gossip.cuh \
		  include/gossip/auxiliary.cuh \
		  include/gossip/context.cuh \
		  include/gossip/all_to_all.cuh \
		  include/gossip/all_to_all_async.cuh \
		  include/gossip/broadcast.cuh \
		  include/gossip/scatter.cuh \
		  include/gossip/gather.cuh \
		  include/gossip/memory_manager.cuh \
		  include/gossip/multisplit.cuh \
		  include/gossip/point_to_point.cuh \
		  include/gossip/transfer_plan.hpp \
		  include/gossip/all_to_all_plan.hpp \
		  include/gossip/broadcast_plan.hpp \
		  include/gossip/scatter_plan.hpp \
		  include/gossip/gather_plan.hpp

DGX = include/gossip/all_to_all_dgx1v.cuh

.PHONY: all clean

all: general dgx1v

general: distributed_general.cu distributed.cuh $(HEADERS) include/plan_parser.cpp include/plan_parser.hpp
	$(NVCC) $(NVCCFLAGS) include/plan_parser.cpp distributed_general.cu -o general

dgx1v: distributed_dgx1v.cu distributed.cuh $(HEADERS) $(DGX)
	$(NVCC) $(NVCCFLAGS) distributed_dgx1v.cu -o dgx1v

test_plans: test_plans.cu $(HEADERS) include/plan_parser.cpp include/plan_parser.hpp
	$(NVCC) $(NVCCFLAGS) include/plan_parser.cpp test_plans.cu -o test_plans

clean:
	rm -rf general dgx1v test_plans

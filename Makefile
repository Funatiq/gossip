NVCC=nvcc
NVCCGENCODE = \
              -gencode arch=compute_60,code=sm_60 \
              -gencode arch=compute_70,code=sm_70

NVCCFLAGS = $(NVCCGENCODE) -O3 -std=c++11 --expt-extended-lambda -Xcompiler="-fopenmp" -Wreorder -lineinfo

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

BUILD_DIR = build


.PHONY: all clean

all: execute simulate


execute: $(BUILD_DIR) $(BUILD_DIR)/plan_parser.o $(BUILD_DIR)/execute.o
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/plan_parser.o $(BUILD_DIR)/execute.o -o execute

$(BUILD_DIR)/execute.o: execute.cu executor.cuh $(HEADERS) include/plan_parser.hpp
	$(NVCC) $(NVCCFLAGS) -c execute.cu -o $(BUILD_DIR)/execute.o


simulate: $(BUILD_DIR) $(BUILD_DIR)/plan_parser.o $(BUILD_DIR)/simulate.o
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/plan_parser.o $(BUILD_DIR)/simulate.o -o simulate

$(BUILD_DIR)/simulate.o: simulate.cu executor.cuh $(HEADERS) include/plan_parser.hpp
	$(NVCC) $(NVCCFLAGS) -c simulate.cu -o $(BUILD_DIR)/simulate.o


$(BUILD_DIR):
	mkdir $(BUILD_DIR)

$(BUILD_DIR)/plan_parser.o: include/plan_parser.cpp include/plan_parser.hpp
	$(NVCC) $(NVCCFLAGS) -c include/plan_parser.cpp -o $(BUILD_DIR)/plan_parser.o


clean:
	rm -rf $(BUILD_DIR)
	rm -rf execute

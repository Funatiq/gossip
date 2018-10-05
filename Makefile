NVCC=nvcc
NVCCFLAGS=-O3 -std=c++14 -arch=sm_70 --expt-extended-lambda -Xcompiler="-fopenmp"

all: general dgx1v

general: distributed_general.cu
	$(NVCC) $(NVCCFLAGS) distributed_general.cu -o general

dgx1v: distributed_dgx1v.cu
	$(NVCC) $(NVCCFLAGS) distributed_dgx1v.cu -o dgx1v

clean:
	rm -rf general dgx1v

#ifndef CUDA_HELPERS_CUH
#define CUDA_HELPERS_CUH
// Created by Christian Hundt [github.com/gravitino]
// Edited  by Daniel Jünger   [github.com/sleeepyjack]

#include <iostream>
#include <cstdint>

#ifndef __CUDACC__
    #include <chrono>
#endif

#ifndef __CUDACC__
    #define TIMERSTART(label)                                                  \
        std::chrono::time_point<std::chrono::system_clock> a##label, b##label; \
        a##label = std::chrono::system_clock::now();
#else
    #define TIMERSTART(label)                                                  \
        cudaEvent_t start##label, stop##label;                                 \
        float time##label;                                                     \
        cudaEventCreate(&start##label);                                        \
        cudaEventCreate(&stop##label);                                         \
        cudaEventRecord(start##label, 0);
#endif

#ifndef __CUDACC__
    #define TIMERSTOP(label)                                                   \
        b##label = std::chrono::system_clock::now();                           \
        std::chrono::duration<double> delta##label = b##label-a##label;        \
        std::cout << "# elapsed time ("<< #label <<"): "                       \
                  << delta##label.count()  << "s" << std::endl;
#else
    #define TIMERSTOP(label)                                                   \
            cudaEventRecord(stop##label, 0);                                   \
            cudaEventSynchronize(stop##label);                                 \
            cudaEventElapsedTime(&time##label, start##label, stop##label);     \
            std::cout << "TIMING: " << time##label << " ms (" << #label << ")" \
                      << std::endl;
#endif

#ifdef __CUDACC__
    #define BANDWIDTHSTART(label)                                              \
        cudaSetDevice(0);                                                      \
        cudaEvent_t start##label, stop##label;                                 \
        float time##label;                                                     \
        cudaEventCreate(&start##label);                                        \
        cudaEventCreate(&stop##label);                                         \
        cudaEventRecord(start##label, 0);

    #define BANDWIDTHSTOP(label, bytes)                                        \
        cudaSetDevice(0);                                                      \
        cudaEventRecord(stop##label, 0);                                       \
        cudaEventSynchronize(stop##label);                                     \
        cudaEventElapsedTime(&time##label, start##label, stop##label);         \
        double bandwidth##label = (bytes)*1000UL/time##label/(1UL<<30);        \
        std::cout << "TIMING: " << time##label << " ms "                       \
                << "-> " << bandwidth##label << " GB/s bandwidth ("    \
                << #label << ")" << std::endl;
#endif


#ifdef __CUDACC__
    #define CUERR {                                                            \
        cudaError_t err;                                                       \
        if ((err = cudaGetLastError()) != cudaSuccess) {                       \
            std::cout << "CUDA error: " << cudaGetErrorString(err) << " : "    \
                      << __FILE__ << ", line " << __LINE__ << std::endl;       \
            exit(1);                                                           \
        }                                                                      \
    }

    // transfer enums
    #define H2D (cudaMemcpyHostToDevice)
    #define D2H (cudaMemcpyDeviceToHost)
    #define H2H (cudaMemcpyHostToHost)
    #define D2D (cudaMemcpyDeviceToDevice)
#endif

// safe division
#define SDIV(x,y)(((x)+(y)-1)/(y))

// cross platform classifiers
#ifdef __CUDACC__
    #define HOSTDEVICEQUALIFIER  __host__ __device__
#else
    #define HOSTDEVICEQUALIFIER
#endif

#ifdef __CUDACC__
    #define INLINEQUALIFIER  __forceinline__
#else
    #define INLINEQUALIFIER inline
#endif

#ifdef __CUDACC__
    #define GLOBALQUALIFIER  __global__
#else
    #define GLOBALQUALIFIER
#endif

#ifdef __CUDACC__
    #define DEVICEQUALIFIER  __device__
#else
    #define DEVICEQUALIFIER
#endif

#ifdef __CUDACC__
    #define HOSTQUALIFIER  __host__
#else
    #define HOSTQUALIFIER
#endif

//helper kernels
#ifdef __CUDACC__

template <typename data_t>
GLOBALQUALIFIER void memset_kernel(data_t * data, size_t capacity, const data_t value)
{
    for (size_t thid = blockDim.x*blockIdx.x+threadIdx.x; thid < capacity; thid += blockDim.x*gridDim.x)
    {
        data[thid] = value;
    }
}

template <typename index_t>
GLOBALQUALIFIER void iota_kernel(index_t * data, size_t capacity)
{
    for (size_t thid = blockDim.x*blockIdx.x+threadIdx.x; thid < capacity; thid += blockDim.x*gridDim.x)
    {
        data[thid] = thid;
    }
}

template<class func_t>
GLOBALQUALIFIER void generic_kernel(func_t f)
{
    f();
}

DEVICEQUALIFIER INLINEQUALIFIER unsigned int lane_id() {
    unsigned int lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}

#if CUDART_VERSION >= 9000
#include <cooperative_groups.h>
template<typename index_t> //index_t either unsigned int or unsigned long long int
DEVICEQUALIFIER INLINEQUALIFIER index_t atomicAggInc(index_t * ctr)
{
    using namespace cooperative_groups;
    coalesced_group g = coalesced_threads();
    index_t prev;
    if (g.thread_rank() == 0) {
        prev = atomicAdd(ctr, g.size());
    }
    prev = g.thread_rank() + g.shfl(prev, 0);
    return prev;
}
#else
template<typename index_t> //index_t either unsigned int or unsigned long long int
DEVICEQUALIFIER INLINEQUALIFIER index_t atomicAggInc(index_t * ctr)
{
    int lane = lane_id();
    //check if thread is active
    int mask = __ballot(1);
    //determine first active lane for atomic add
    int leader = __ffs(mask) - 1;
    index_t res;
    if (lane == leader) res = atomicAdd(ctr, __popc(mask));
    //broadcast to warp
    res = __shfl(res, leader);
    //compute index for each thread
    return res + __popc(mask & ((1 << lane) -1));
}

//FIXME this hack is so dirrrty x-tina would be proud
namespace cooperative_groups
{

    DEVICEQUALIFIER INLINEQUALIFIER
    int this_thread_block()
    {
        return 0; //not needed
    }

    template<unsigned int Size>
    class tiled_partition
    {
        const unsigned int mask = ((1ULL<<Size)-1)<<((Size*threadIdx.y)%32);
    public:

        DEVICEQUALIFIER
        tiled_partition(int ignore)
        {
            
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        unsigned int size() const
        {
            return Size;
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        unsigned int thread_rank() const
        {
            return threadIdx.x;
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        unsigned int ballot(bool pred) const
        {
            return ((mask & __ballot(pred)) >> (Size*threadIdx.y));
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        bool any(bool pred) const
        {
            return (ballot(pred) != 0);
        }

        DEVICEQUALIFIER INLINEQUALIFIER
        bool all(bool pred) const
        {
            return (__popc(ballot(pred)) == Size);
        }
    };


} //cooperative groups

#endif

#endif

#endif /*CUDA_HELPERS_CUH*/

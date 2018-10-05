#pragma once

template <
    uint64_t num_gpus>
struct part_hash {
    template <
        typename index_t> __host__ __device__ __forceinline__
    uint64_t operator()(index_t x) const {
        return uint64_t(x) % (num_gpus+1);
    }
};
/*
template <
    typename index_t> __device__ __forceinline__
index_t atomicAggInc(
    index_t * counter) {

    // accumulate over whole warp to reduce atomic congestion
    const int lane = threadIdx.x % 32;
    const int mask = __ballot(1);// __activemask();
    const int leader = __ffs(mask) - 1;
    index_t res;
    if (lane == leader)
        res = atomicAdd(counter, __popc(mask));
    res = __shfl(res, leader); //__shfl_sync(0xFFFFFFFF, res, leader);

    return res + __popc(mask & ((1 << lane) -1));
}
*/

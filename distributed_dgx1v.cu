#include <iostream>
#include <cstdint>
#include "include/gossip.cuh"

template <typename data_t>
__global__
void memset_kernel(data_t * data, size_t capacity, const data_t value)
{
    for (size_t thid = blockDim.x*blockIdx.x+threadIdx.x; thid < capacity; thid += blockDim.x*gridDim.x)
    {
        data[thid] = value;
    }
}

template <
    typename value_t,
    typename index_t,
    typename funct_t> __global__
void validate(
    value_t const * const data,
    index_t const length,
    index_t const device_id,
    funct_t const predicate) {

    const uint64_t thid = blockDim.x*blockIdx.x+threadIdx.x;

    for (uint64_t i = thid; i < length; i += blockDim.x*gridDim.x)
        if(predicate(data[i]) != device_id)
            printf("ERROR on gpu %lu at index %lu: %lu with predicate %lu \n",
                    uint64_t(device_id-1), i, uint64_t(data[i]), uint64_t(data[i]) % 8);

}

#define BIG_CONSTANT(x) (x##LLU)
__host__ __device__ uint64_t fmix64(uint64_t k) {
	k ^= k >> 33;
	k *= BIG_CONSTANT(0xff51afd7ed558ccd);
	k ^= k >> 33;
	k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
	k ^= k >> 33;

	return k;
}

int main () {

    typedef uint64_t data_t;
    constexpr uint64_t num_gpus = 8;
    double security_factor = 1.5;

    uint64_t batch_size = 1UL << 29;
    uint64_t batch_size_secure = batch_size * security_factor;
    uint64_t device_ids[num_gpus] = {0, 1, 2, 3, 4, 5, 6, 7};

    auto context = new gossip::context_t<num_gpus>(device_ids);
    auto all2all = new gossip::all2all_dgx1v_t<num_gpus>(context);
    auto multisplit = new gossip::multisplit_t<num_gpus>(context);
    auto point2point = new gossip::point2point_t<num_gpus>(context);

    data_t * ying[num_gpus];
    data_t * yang[num_gpus];

    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context->get_device_id(gpu));
        cudaMalloc(&ying[gpu], sizeof(data_t)*batch_size_secure);
        cudaMalloc(&yang[gpu], sizeof(data_t)*batch_size_secure);
    } CUERR

    context->sync_all_streams();
    TIMERSTART(zero_gpu_buffers)
    const data_t init_data = 0;
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu){
        cudaSetDevice(context->get_device_id(gpu));
        memset_kernel<<<256, 1024, 0,
            context->get_streams(gpu)[0 % num_gpus]>>>
            (ying[gpu], batch_size_secure, init_data);
        memset_kernel<<<256, 1024, 0,
            context->get_streams(gpu)[1 % num_gpus]>>>
            (yang[gpu], batch_size_secure, init_data);
    } CUERR

    context->sync_all_streams();
    TIMERSTOP(zero_gpu_buffers)

    TIMERSTART(malloc_host)
    data_t * data_h = nullptr;
    cudaMallocHost(&data_h, sizeof(data_t)*batch_size*num_gpus); CUERR
    TIMERSTOP(malloc_host)

    std::cout << "INFO: " << sizeof(data_t)*batch_size*num_gpus << " bytes" << std::endl;

    TIMERSTART(init_host_data)
    # pragma omp parallel for
    for (uint64_t i = 0; i < batch_size*num_gpus; i++)
        data_h[i] = fmix64(i+1);
    TIMERSTOP(init_host_data)

    // this array partitions the widthcontext->get_device_id(gpu) many elements into
    // num_gpus many portions of approximately equal size
    data_t * srcs[num_gpus] = {nullptr};
    data_t * dsts[num_gpus] = {nullptr};
    uint64_t lens[num_gpus] = {0};

    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        const uint64_t lower = gpu*batch_size;
        const uint64_t upper = lower+batch_size;

        srcs[gpu] = data_h+lower;
        dsts[gpu] = ying[gpu];
        lens[gpu] = upper-lower;
    }

    TIMERSTART(H2D_async)
    // move batches to buffer ying
    point2point->execH2DAsync(srcs, dsts, lens);
    point2point->sync();
    TIMERSTOP(H2D_async)

    // perform multisplit on each device
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs[gpu] = ying[gpu];
        dsts[gpu] = yang[gpu];
    }
    auto part_hash = [=] DEVICEQUALIFIER (const data_t& x){
        return (x % num_gpus) + 1;
    };

    TIMERSTART(multisplit)
    uint64_t table[num_gpus][num_gpus];
    multisplit->execAsync(srcs, lens, dsts, lens, table, part_hash);
    multisplit->sync();
    TIMERSTOP(multisplit)

    std::cout << std::endl << "Partition Table:" << std::endl;
    for (uint64_t src = 0; src < num_gpus; src++)
        for (uint64_t trg = 0; trg < num_gpus; trg++)
            std::cout << table[src][trg] << (trg+1 == num_gpus ? "\n" : " ");
    std::cout << std::endl;

    // perform all to all communication
    uint64_t srcs_lens[num_gpus];
    uint64_t dsts_lens[num_gpus];
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs_lens[gpu] = batch_size_secure;
        dsts_lens[gpu] = batch_size_secure;
        srcs[gpu] = yang[gpu];
        dsts[gpu] = ying[gpu];
    }

    TIMERSTART(all2all)
    all2all->execAsync(srcs, srcs_lens, dsts, dsts_lens, table);
    all2all->sync();
    TIMERSTOP(all2all)

    TIMERSTART(validate)
    uint64_t lengths[num_gpus];
    for (uint64_t trg = 0; trg < num_gpus; trg++) {
        lengths[trg] = 0;
        for (uint64_t src = 0; src < num_gpus; src++)
            lengths[trg] += table[src][trg];
    }

    for (uint64_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(context->get_device_id(gpu));
        validate<<<256, 1024, 0, context->get_streams(gpu)[0]>>>
            (srcs[gpu], lengths[gpu], context->get_device_id(gpu)+1, part_hash);
    }
    CUERR
    TIMERSTOP(validate)

    context->sync_hard();
    for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context->get_device_id(gpu)); CUERR
        cudaFree(ying[gpu]);        CUERR
        cudaFree(yang[gpu]);        CUERR
    } CUERR

    cudaFreeHost(data_h);           CUERR

    context->sync_hard();
    delete all2all;
    delete multisplit;
    delete point2point;
    delete context;
}

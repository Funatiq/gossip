#include <iostream>
#include <cstdint>
#include "include/gossip.cuh"
#include "include/cudahelpers/cuda_helpers.cuh"

using gpu_id_t = gossip::gpu_id_t;

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
    typename gpu_id_t,
    typename funct_t> __global__
void validate(
    value_t const * const data,
    index_t const length,
    gpu_id_t const device_id,
    funct_t const predicate) {

    const uint64_t thid = blockDim.x*blockIdx.x+threadIdx.x;

    for (uint64_t i = thid; i < length; i += blockDim.x*gridDim.x)
        if(predicate(data[i]) != device_id)
            printf("ERROR on gpu %lu at index %lu: %lu with predicate %lu \n",
                    uint64_t(device_id-1), i, uint64_t(data[i]), predicate(data[i]));

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

template<
    typename data_t,
    class T1,
    class T2,
    class T3,
    class T4>
void run_multisplit_all2all(
        T1* context,
        T2* all2all,
        T3* multisplit,
        T4* point2point,
        const size_t batch_size,
        const size_t batch_size_secure)
{
    gpu_id_t num_gpus = context->get_num_devices();

    std::vector<data_t *> ying(num_gpus);
    std::vector<data_t *> yang(num_gpus);
    std::vector<data_t *> yong(num_gpus);

    TIMERSTART(malloc_devices)
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context->get_device_id(gpu));
        cudaMalloc(&ying[gpu], sizeof(data_t)*batch_size_secure);
        cudaMalloc(&yang[gpu], sizeof(data_t)*batch_size_secure);
        cudaMalloc(&yong[gpu], sizeof(data_t)*batch_size_secure);
    } CUERR
    context->sync_hard();
    TIMERSTOP(malloc_devices)

    context->sync_all_streams();

    TIMERSTART(zero_gpu_buffers)
    const data_t init_data = 0;
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu){
        cudaSetDevice(context->get_device_id(gpu));
        memset_kernel<<<256, 1024, 0,
            context->get_streams(gpu)[0 % num_gpus]>>>
            (ying[gpu], batch_size_secure, init_data);
        memset_kernel<<<256, 1024, 0,
            context->get_streams(gpu)[1 % num_gpus]>>>
            (yang[gpu], batch_size_secure, init_data);
        memset_kernel<<<256, 1024, 0,
            context->get_streams(gpu)[1 % num_gpus]>>>
            (yong[gpu], batch_size_secure, init_data);
    }
    context->sync_all_streams();
    CUERR
    TIMERSTOP(zero_gpu_buffers)

    std::cout << "INFO: " << sizeof(data_t)*batch_size*num_gpus << " bytes (all2all)" << std::endl;

    TIMERSTART(malloc_host)
    data_t * data_h = nullptr;
    cudaMallocHost(&data_h, sizeof(data_t)*batch_size*num_gpus); CUERR
    TIMERSTOP(malloc_host)

    TIMERSTART(init_host_data)
    # pragma omp parallel for
    for (size_t i = 0; i < batch_size*num_gpus; i++)
        data_h[i] = fmix64(i+1);
    TIMERSTOP(init_host_data)

    // this array partitions the widthcontext->get_device_id(gpu) many elements into
    // num_gpus many portions of approximately equal size
    std::vector<data_t *> srcs(num_gpus);
    std::vector<data_t *> dsts(num_gpus);
    std::vector<size_t  > lens(num_gpus);

    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        const size_t lower = gpu*batch_size;
        const size_t upper = lower+batch_size;

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
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs[gpu] = ying[gpu];
        dsts[gpu] = yang[gpu];
    }
    auto part_hash = [=] DEVICEQUALIFIER (const data_t& x){
        return (x % num_gpus) + 1;
    };

    std::vector<std::vector<size_t>> table(num_gpus, std::vector<size_t>(num_gpus));
    TIMERSTART(multisplit)
    multisplit->execAsync(srcs, lens, dsts, lens, table, part_hash);
    multisplit->sync();
    TIMERSTOP(multisplit)

    std::cout << std::endl << "Partition Table:" << std::endl;
    for (gpu_id_t src = 0; src < num_gpus; src++)
        for (gpu_id_t trg = 0; trg < num_gpus; trg++)
            std::cout << table[src][trg] << (trg+1 == num_gpus ? "\n" : " ");
    std::cout << std::endl;

    // perform all to all communication
    std::vector<size_t> srcs_lens(num_gpus);
    std::vector<size_t> dsts_lens(num_gpus);
    std::vector<data_t *> bufs(num_gpus);
    std::vector<size_t> bufs_lens(num_gpus);
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs_lens[gpu] = batch_size_secure;
        dsts_lens[gpu] = batch_size_secure;
        bufs_lens[gpu] = batch_size_secure;
        srcs[gpu] = yang[gpu];
        dsts[gpu] = ying[gpu];
        bufs[gpu] = yong[gpu];
    }

    // all2all->show_plan();

    TIMERSTART(all2all)
    all2all->execAsync(srcs, srcs_lens, dsts, dsts_lens, bufs, bufs_lens, table);
    all2all->sync();
    TIMERSTOP(all2all)

    TIMERSTART(validate)
    std::vector<size_t> lengths(num_gpus);
    for (gpu_id_t trg = 0; trg < num_gpus; trg++) {
        lengths[trg] = 0;
        for (gpu_id_t src = 0; src < num_gpus; src++)
            lengths[trg] += table[src][trg];
    }

    for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(context->get_device_id(gpu));
        validate<<<256, 1024, 0, context->get_streams(gpu)[0]>>>
            (dsts[gpu], lengths[gpu], gpu+1, part_hash);
    }
    CUERR
    TIMERSTOP(validate)

    context->sync_hard();
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context->get_device_id(gpu)); CUERR
        cudaFree(ying[gpu]);        CUERR
        cudaFree(yang[gpu]);        CUERR
    } CUERR

    cudaFreeHost(data_h);           CUERR
}

template<
    typename data_t,
    class T1,
    class T2,
    class T3,
    class T4,
    class T5>
void run_multisplit_scatter_gather(
        T1* context,
        T2* point2point,
        T3* multisplit,
        T4* scatter,
        T5* gather,
        const size_t batch_size,
        const size_t batch_size_secure)
{
    gpu_id_t num_gpus = context->get_num_devices();
    gpu_id_t main_gpu = 0;

    std::vector<data_t *> ying(num_gpus);
    std::vector<data_t *> yang(num_gpus);

    TIMERSTART(malloc_devices)
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context->get_device_id(gpu));
        cudaMalloc(&ying[gpu], sizeof(data_t)*batch_size_secure);
        cudaMalloc(&yang[gpu], sizeof(data_t)*batch_size_secure);
    } CUERR
    context->sync_hard();
    TIMERSTOP(malloc_devices)

    context->sync_all_streams();

    TIMERSTART(zero_gpu_buffers)
    const data_t init_data = 0;
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu){
        cudaSetDevice(context->get_device_id(gpu));
        memset_kernel<<<256, 1024, 0,
            context->get_streams(gpu)[0 % num_gpus]>>>
            (ying[gpu], batch_size_secure, init_data);
        memset_kernel<<<256, 1024, 0,
            context->get_streams(gpu)[1 % num_gpus]>>>
            (yang[gpu], batch_size_secure, init_data);
    }
    context->sync_all_streams();
    CUERR
    TIMERSTOP(zero_gpu_buffers)

    std::cout << "INFO: " << sizeof(data_t)*batch_size << " bytes (scatter/gather)" << std::endl;

    TIMERSTART(malloc_host)
    data_t * data_h = nullptr;
    cudaMallocHost(&data_h, sizeof(data_t)*batch_size); CUERR
    TIMERSTOP(malloc_host)

    TIMERSTART(init_host_data)
    # pragma omp parallel for
    for (size_t i = 0; i < batch_size; i++)
        data_h[i] = fmix64(i+1);
    TIMERSTOP(init_host_data)

    std::vector<data_t *> srcs(num_gpus);
    std::vector<data_t *> dsts(num_gpus);
    std::vector<size_t  > lens(num_gpus);

    // copy batch to main device
    // all other devices get nothing
    lens[main_gpu] = batch_size;

    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs[gpu] = data_h;
        dsts[gpu] = ying[gpu];
    }

    TIMERSTART(H2D_async)
    // move batches to buffer ying
    point2point->execH2DAsync(srcs, dsts, lens);
    point2point->sync();
    TIMERSTOP(H2D_async)


    // perform multisplit on main device
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs[gpu] = ying[gpu];
        dsts[gpu] = yang[gpu];
    }
    auto part_hash = [=] HOSTDEVICEQUALIFIER (const data_t& x){
        return (x % num_gpus) + 1;
    };

    std::vector<std::vector<size_t>> table(num_gpus, std::vector<size_t>(num_gpus));
    TIMERSTART(multisplit)
    multisplit->execAsync(srcs, lens, dsts, lens, table, part_hash);
    multisplit->sync();
    TIMERSTOP(multisplit)

    std::cout << std::endl << "Partition Table:" << std::endl;
    for (gpu_id_t src = 0; src < num_gpus; src++)
        for (gpu_id_t trg = 0; trg < num_gpus; trg++)
            std::cout << table[src][trg] << (trg+1 == num_gpus ? "\n" : " ");
    std::cout << std::endl;


    // perform scatter
    std::vector<size_t> mems_lens(num_gpus);
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs[gpu] = yang[gpu];
        dsts[gpu] = ying[gpu];
        mems_lens[gpu] = batch_size_secure;
    }

    // scatter->show_plan();

    TIMERSTART(scatter)
    scatter->execAsync(srcs[main_gpu], batch_size_secure, table[main_gpu], dsts, mems_lens);
    scatter->sync();
    TIMERSTOP(scatter)

    TIMERSTART(validate_scatter)
    for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(context->get_device_id(gpu));
        validate<<<256, 1024, 0, context->get_streams(gpu)[0]>>>
            (dsts[gpu], table[main_gpu][gpu], gpu+1, part_hash);
    }
    context->sync_hard();
    CUERR
    TIMERSTOP(validate_scatter)

    std::cout << '\n';


    TIMERSTART(clear_gpu_buffers)
    const data_t clear_data = data_t(-1);
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu){
        cudaSetDevice(context->get_device_id(gpu));
        memset_kernel<<<256, 1024, 0,
            context->get_streams(gpu)[0 % num_gpus]>>>
            (srcs[gpu], batch_size_secure, clear_data);
    }
    context->sync_all_streams();
    CUERR
    TIMERSTOP(clear_gpu_buffers)


    // perform gather
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs[gpu] = ying[gpu];
        dsts[gpu] = yang[gpu];
    }

    // gather->show_plan();

    TIMERSTART(gather)
    gather->execAsync(srcs, mems_lens, table[main_gpu], dsts[main_gpu], batch_size_secure);
    gather->sync();
    TIMERSTOP(gather)

    TIMERSTART(validate_gather)
    std::vector<data_t *> mems2(num_gpus, dsts[main_gpu]);
    for (gpu_id_t trg = 1; trg < num_gpus; trg++) {
        mems2[trg] = mems2[trg-1] + table[main_gpu][trg-1];
    }

    cudaSetDevice(context->get_device_id(main_gpu));
    for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
        validate<<<256, 1024, 0, context->get_streams(main_gpu)[0]>>>
            (mems2[gpu], table[main_gpu][gpu], gpu+1, part_hash);
    }
    context->sync_hard();
    CUERR
    TIMERSTOP(validate_gather)

    context->sync_hard();
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context->get_device_id(gpu)); CUERR
        cudaFree(ying[gpu]);        CUERR
        cudaFree(yang[gpu]);        CUERR
    } CUERR

    cudaFreeHost(data_h);           CUERR
}

#pragma once

#include <iostream>
#include <cstdint>
#include "include/gossip.cuh"
#include "include/cudahelpers/cuda_helpers.cuh"

using gpu_id_t = gossip::gpu_id_t;

#define BIG_CONSTANT(x) (x##LLU)
__host__ __device__ uint64_t fmix64(uint64_t k) {
	k ^= k >> 33;
	k *= BIG_CONSTANT(0xff51afd7ed558ccd);
	k ^= k >> 33;
	k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
	k ^= k >> 33;

	return k;
}

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
    typename index_t> __global__
void generate_data(
    value_t * data,
    index_t const length,
    index_t const offset
) {
    const uint64_t thid = blockDim.x*blockIdx.x+threadIdx.x;

    for (uint64_t i = thid; i < length; i += blockDim.x*gridDim.x) {
        data[i] = fmix64(offset+i+1);
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

template<
    typename data_t,
    class T1,
    class T2,
    class T3,
    class T4>
void run_multisplit_all2all(
        T1& context,
        T2& all2all,
        T3& multisplit,
        T4& point2point,
        const size_t batch_size,
        const size_t batch_size_secure)
{
    gpu_id_t num_gpus = context.get_num_devices();

    std::vector<data_t *> ying(num_gpus);
    std::vector<data_t *> yang(num_gpus);

    TIMERSTART(malloc_devices)
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu));
        cudaMalloc(&ying[gpu], sizeof(data_t)*batch_size_secure);
        cudaMalloc(&yang[gpu], sizeof(data_t)*batch_size_secure);
    } CUERR
    context.sync_hard();
    TIMERSTOP(malloc_devices)

    TIMERSTART(zero_gpu_buffers)
    const data_t init_data = 0;
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu){
        cudaSetDevice(context.get_device_id(gpu));
        memset_kernel<<<256, 1024, 0,
            context.get_streams(gpu)[0 % num_gpus]>>>
            (ying[gpu], batch_size_secure, init_data);
        memset_kernel<<<256, 1024, 0,
            context.get_streams(gpu)[1 % num_gpus]>>>
            (yang[gpu], batch_size_secure, init_data);
    }
    context.sync_all_streams();
    CUERR
    TIMERSTOP(zero_gpu_buffers)

    std::cout << "INFO: " << sizeof(data_t)*batch_size*num_gpus << " bytes (all2all)" << std::endl;

    std::vector<data_t *> srcs(num_gpus);
    std::vector<data_t *> dsts(num_gpus);
    std::vector<size_t  > lens(num_gpus);

    // generate batch of data on each device
    TIMERSTART(init_data)
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs[gpu] = ying[gpu];
        dsts[gpu] = yang[gpu];
        lens[gpu] = batch_size;

        cudaSetDevice(context.get_device_id(gpu));
        generate_data<<<256, 1024, 0, context.get_streams(gpu)[0]>>>
            (srcs[gpu], lens[gpu], gpu*batch_size);
    }
    context.sync_all_streams();
    CUERR
    TIMERSTOP(init_data)

    // perform multisplit on each device
    auto part_hash = [=] DEVICEQUALIFIER (const data_t& x){
        return (x % num_gpus);
    };

    std::vector<std::vector<size_t>> table(num_gpus, std::vector<size_t>(num_gpus));
    TIMERSTART(multisplit)
    multisplit.execAsync(srcs, lens, dsts, lens, table, part_hash);
    multisplit.sync();
    TIMERSTOP(multisplit)

    std::cout << std::endl << "Partition Table:" << std::endl;
    for (gpu_id_t src = 0; src < num_gpus; src++)
        for (gpu_id_t trg = 0; trg < num_gpus; trg++)
            std::cout << table[src][trg] << (trg+1 == num_gpus ? "\n" : " ");
    std::cout << std::endl;

    // reset srcs to zero
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu));
        cudaMemsetAsync(srcs[gpu], 0, sizeof(data_t)*lens[gpu],
                        context.get_streams(gpu)[0]);
    }
    context.sync_all_streams();
    CUERR

    // perform all to all communication
    std::vector<size_t> srcs_lens(num_gpus);
    std::vector<size_t> dsts_lens(num_gpus);
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs_lens[gpu] = batch_size_secure;
        dsts_lens[gpu] = batch_size_secure;
        srcs[gpu] = yang[gpu];
        dsts[gpu] = ying[gpu];
    }

    // all2all.show_plan();

    TIMERSTART(all2all)
    all2all.execAsync(srcs, srcs_lens, dsts, dsts_lens, table);
    all2all.sync();
    TIMERSTOP(all2all)

    TIMERSTART(validate)
    std::vector<size_t> lengths(num_gpus);
    for (gpu_id_t trg = 0; trg < num_gpus; trg++) {
        lengths[trg] = 0;
        for (gpu_id_t src = 0; src < num_gpus; src++)
            lengths[trg] += table[src][trg];
    }

    for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(context.get_device_id(gpu));
        validate<<<256, 1024, 0, context.get_streams(gpu)[0]>>>
            (dsts[gpu], lengths[gpu], gpu, part_hash);
    }
    CUERR
    TIMERSTOP(validate)

    context.sync_hard();
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu)); CUERR
        cudaFree(ying[gpu]);        CUERR
        cudaFree(yang[gpu]);        CUERR
    } CUERR
}

template<
    typename data_t,
    class T1,
    class T2,
    class T3,
    class T4>
void run_multisplit_all2all_async(
        T1& context,
        T2& all2all,
        T3& multisplit,
        T4& point2point,
        const size_t batch_size,
        const size_t batch_size_secure)
{
    gpu_id_t num_gpus = context.get_num_devices();

    std::vector<data_t *> ying(num_gpus);
    std::vector<data_t *> yang(num_gpus);

    TIMERSTART(malloc_devices)
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu));
        cudaMalloc(&ying[gpu], sizeof(data_t)*batch_size_secure);
        cudaMalloc(&yang[gpu], sizeof(data_t)*batch_size_secure);
    } CUERR
    context.sync_hard();
    TIMERSTOP(malloc_devices)

    context.sync_all_streams();

    TIMERSTART(zero_gpu_buffers)
    const data_t init_data = 0;
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu){
        cudaSetDevice(context.get_device_id(gpu));
        memset_kernel<<<256, 1024, 0,
            context.get_streams(gpu)[0 % num_gpus]>>>
            (ying[gpu], batch_size_secure, init_data);
        memset_kernel<<<256, 1024, 0,
            context.get_streams(gpu)[1 % num_gpus]>>>
            (yang[gpu], batch_size_secure, init_data);
    }
    context.sync_all_streams();
    CUERR
    TIMERSTOP(zero_gpu_buffers)

    std::cout << "INFO: " << sizeof(data_t)*batch_size*num_gpus << " bytes (all2all_async)" << std::endl;

    std::vector<data_t *> srcs(num_gpus);
    std::vector<data_t *> dsts(num_gpus);
    std::vector<size_t  > lens(num_gpus);

    // generate batch of data on each device
    TIMERSTART(init_data)
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs[gpu] = ying[gpu];
        dsts[gpu] = yang[gpu];
        lens[gpu] = batch_size;

        cudaSetDevice(context.get_device_id(gpu));
        generate_data<<<256, 1024, 0, context.get_streams(gpu)[0]>>>
            (srcs[gpu], lens[gpu], gpu*batch_size);
    }
    context.sync_all_streams();
    CUERR
    TIMERSTOP(init_data)

    // perform multisplit on each device
    auto part_hash = [=] DEVICEQUALIFIER (const data_t& x){
        return (x % num_gpus);
    };

    std::vector<std::vector<size_t>> table(num_gpus, std::vector<size_t>(num_gpus));
    TIMERSTART(multisplit)
    multisplit.execAsync(srcs, lens, dsts, lens, table, part_hash);
    multisplit.sync();
    TIMERSTOP(multisplit)

    std::cout << "\nPartition Table:" << std::endl;
    for (gpu_id_t src = 0; src < num_gpus; src++)
        for (gpu_id_t trg = 0; trg < num_gpus; trg++)
            std::cout << table[src][trg] << (trg+1 == num_gpus ? "\n" : " ");
    std::cout << std::endl;

    std::cout << "Required buffer sizes:" << std::endl;
    std::vector<size_t> bufs_lens_calc = all2all.calcBufferLengths(table);
    for (const auto& buf_len_calc : bufs_lens_calc) {
        std::cout << buf_len_calc << ' ';
    }
    std::cout << '\n' << std::endl;

    std::vector<data_t *> bufs(num_gpus);
    TIMERSTART(malloc_buffers)
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu));
        cudaMalloc(&bufs[gpu], sizeof(data_t)*bufs_lens_calc[gpu]);
    } CUERR
    context.sync_hard();
    TIMERSTOP(malloc_buffers)

    // prepare all to all communication
    std::vector<size_t> srcs_lens(num_gpus);
    std::vector<size_t> dsts_lens(num_gpus);
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs_lens[gpu] = batch_size_secure;
        dsts_lens[gpu] = batch_size_secure;
        srcs[gpu] = yang[gpu];
        dsts[gpu] = ying[gpu];
    }

    // reset dsts and buffer to zero
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu));
        cudaMemsetAsync(dsts[gpu], 0, sizeof(data_t)*dsts_lens[gpu],
                        context.get_streams(gpu)[0]);
        cudaMemsetAsync(bufs[gpu], 0, sizeof(data_t)*bufs_lens_calc[gpu],
                        context.get_streams(gpu)[0]);
    }
    context.sync_all_streams();
    CUERR

    // all2all.show_plan();

    TIMERSTART(all2all_async)
    all2all.execAsync(srcs, srcs_lens, dsts, dsts_lens, bufs, bufs_lens_calc, table);
    all2all.sync();
    TIMERSTOP(all2all_async)

    TIMERSTART(validate)
    std::vector<size_t> lengths(num_gpus);
    for (gpu_id_t trg = 0; trg < num_gpus; trg++) {
        lengths[trg] = 0;
        for (gpu_id_t src = 0; src < num_gpus; src++)
            lengths[trg] += table[src][trg];
    }

    for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(context.get_device_id(gpu));
        validate<<<256, 1024, 0, context.get_streams(gpu)[0]>>>
            (dsts[gpu], lengths[gpu], gpu, part_hash);
    }
    CUERR
    TIMERSTOP(validate)

    context.sync_hard();
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu)); CUERR
        cudaFree(ying[gpu]);        CUERR
        cudaFree(yang[gpu]);        CUERR
        cudaFree(bufs[gpu]);        CUERR
    } CUERR
}

template<
    typename data_t,
    class T1,
    class T2,
    class T3,
    class T4,
    class T5>
void run_multisplit_scatter_gather(
        T1& context,
        T2& point2point,
        T3& multisplit,
        T4& scatter,
        T5& gather,
        gpu_id_t main_gpu,
        const size_t batch_size,
        const size_t batch_size_secure)
{
    gpu_id_t num_gpus = context.get_num_devices();

    std::vector<data_t *> ying(num_gpus);
    std::vector<data_t *> yang(num_gpus);

    TIMERSTART(malloc_devices)
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu));
        cudaMalloc(&ying[gpu], sizeof(data_t)*batch_size_secure);
        cudaMalloc(&yang[gpu], sizeof(data_t)*batch_size_secure);
    } CUERR
    context.sync_hard();
    TIMERSTOP(malloc_devices)

    context.sync_all_streams();

    TIMERSTART(zero_gpu_buffers)
    const data_t init_data = 0;
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu){
        cudaSetDevice(context.get_device_id(gpu));
        memset_kernel<<<256, 1024, 0,
            context.get_streams(gpu)[0 % num_gpus]>>>
            (ying[gpu], batch_size_secure, init_data);
        memset_kernel<<<256, 1024, 0,
            context.get_streams(gpu)[1 % num_gpus]>>>
            (yang[gpu], batch_size_secure, init_data);
    }
    context.sync_all_streams();
    CUERR
    TIMERSTOP(zero_gpu_buffers)

    std::cout << "INFO: " << sizeof(data_t)*batch_size << " bytes (scatter_gather)" << std::endl;

    std::vector<data_t *> srcs(num_gpus);
    std::vector<data_t *> dsts(num_gpus);
    std::vector<size_t  > lens(num_gpus);

    // generate batch of data on main device
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs[gpu] = ying[gpu];
        dsts[gpu] = yang[gpu];
    }
    lens[main_gpu] = batch_size;

    TIMERSTART(init_data)
    cudaSetDevice(context.get_device_id(main_gpu));
    generate_data<<<256, 1024, 0, context.get_streams(main_gpu)[0]>>>
        (srcs[main_gpu], lens[main_gpu], size_t(0));
    context.sync_all_streams();
    CUERR
    TIMERSTOP(init_data)

    // perform multisplit on main device
    auto part_hash = [=] HOSTDEVICEQUALIFIER (const data_t& x){
        return (x % num_gpus);
    };

    std::vector<std::vector<size_t>> table(num_gpus, std::vector<size_t>(num_gpus));
    TIMERSTART(multisplit)
    multisplit.execAsync(srcs, lens, dsts, lens, table, part_hash);
    multisplit.sync();
    TIMERSTOP(multisplit)

    std::cout << "\nPartition Table:" << std::endl;
    for (gpu_id_t src = 0; src < num_gpus; src++)
        for (gpu_id_t trg = 0; trg < num_gpus; trg++)
            std::cout << table[src][trg] << (trg+1 == num_gpus ? "\n" : " ");
    std::cout << std::endl;

    std::cout << "Required buffer sizes:" << std::endl;
    std::vector<size_t> bufs_lens_calc_scatter = scatter.calcBufferLengths(table[main_gpu]);
    for (const auto& buf_len_calc : bufs_lens_calc_scatter) {
        std::cout << buf_len_calc << ' ';
    }
    std::cout << '\n' << std::endl;

    std::vector<data_t *> bufs(num_gpus);
    std::vector<size_t> bufs_lens(bufs_lens_calc_scatter);
    TIMERSTART(malloc_buffers)
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu));
        cudaMalloc(&bufs[gpu], sizeof(data_t)*bufs_lens[gpu]);
    } CUERR
    context.sync_hard();
    TIMERSTOP(malloc_buffers)

    // prepare scatter
    std::vector<size_t> mems_lens(num_gpus);
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs[gpu] = yang[gpu];
        dsts[gpu] = ying[gpu];
        mems_lens[gpu] = batch_size_secure;
    }

    // reset dsts and buffer to zero
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu));
        cudaMemsetAsync(dsts[gpu], 0, sizeof(data_t)*mems_lens[gpu],
                        context.get_streams(gpu)[0]);
        cudaMemsetAsync(bufs[gpu], 0, sizeof(data_t)*bufs_lens[gpu],
                        context.get_streams(gpu)[0]);
    }
    context.sync_all_streams();
    CUERR

    // scatter.show_plan();

    TIMERSTART(scatter)
    scatter.execAsync(srcs[main_gpu], batch_size_secure,
                      dsts, mems_lens,
                      bufs, bufs_lens,
                      table[main_gpu]);
    scatter.sync();
    TIMERSTOP(scatter)

    TIMERSTART(validate_scatter)
    for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(context.get_device_id(gpu));
        validate<<<256, 1024, 0, context.get_streams(gpu)[0]>>>
            (dsts[gpu], table[main_gpu][gpu], gpu, part_hash);
    }
    context.sync_hard();
    CUERR
    TIMERSTOP(validate_scatter)

    std::cout << '\n';


    std::cout << "Required buffer sizes:" << std::endl;
    std::vector<size_t> bufs_lens_calc_gather = gather.calcBufferLengths(table[main_gpu]);
    for (const auto& buf_len_calc : bufs_lens_calc_gather) {
        std::cout << buf_len_calc << ' ';
    }
    std::cout << '\n' << std::endl;

    TIMERSTART(realloc_buffers)
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        if(bufs_lens[gpu] < bufs_lens_calc_gather[gpu]) {
            bufs_lens[gpu] = bufs_lens_calc_gather[gpu];
            cudaSetDevice(context.get_device_id(gpu));
            cudaFree(bufs[gpu]);
            cudaMalloc(&bufs[gpu], sizeof(data_t)*bufs_lens[gpu]);
        }
    } CUERR
    context.sync_hard();
    TIMERSTOP(realloc_buffers)

    // prepare gather
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs[gpu] = ying[gpu];
        dsts[gpu] = yang[gpu];
    }

    // reset dsts and buffer to zero
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu));
        cudaMemsetAsync(dsts[gpu], ~0, sizeof(data_t)*mems_lens[gpu],
                        context.get_streams(gpu)[0]);
        cudaMemsetAsync(bufs[gpu], ~0, sizeof(data_t)*bufs_lens[gpu],
                        context.get_streams(gpu)[0]);
    }
    context.sync_all_streams();
    CUERR

    // gather.show_plan();

    TIMERSTART(gather)
    gather.execAsync(srcs, mems_lens,
                     dsts[main_gpu], batch_size_secure,
                     bufs, bufs_lens,
                     table[main_gpu]);
    gather.sync();
    TIMERSTOP(gather)

    TIMERSTART(validate_gather)
    std::vector<data_t *> mems2(num_gpus, dsts[main_gpu]);
    for (gpu_id_t trg = 1; trg < num_gpus; trg++) {
        mems2[trg] = mems2[trg-1] + table[main_gpu][trg-1];
    }

    cudaSetDevice(context.get_device_id(main_gpu));
    for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
        validate<<<256, 1024, 0, context.get_streams(main_gpu)[0]>>>
            (mems2[gpu], table[main_gpu][gpu], gpu, part_hash);
    }
    context.sync_hard();
    CUERR
    TIMERSTOP(validate_gather)

    context.sync_hard();
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu)); CUERR
        cudaFree(ying[gpu]);        CUERR
        cudaFree(yang[gpu]);        CUERR
        cudaFree(bufs[gpu]);        CUERR
    } CUERR
}

template<
    typename data_t,
    class T1,
    class T2,
    class T3,
    class T4>
void run_multisplit_broadcast(
        T1& context,
        T2& point2point,
        T3& multisplit,
        T4& broadcast,
        const size_t batch_size,
        const size_t batch_size_secure)
{
    gpu_id_t num_gpus = context.get_num_devices();
    gpu_id_t main_gpu = 0;

    std::vector<data_t *> ying(num_gpus);
    std::vector<data_t *> yang(num_gpus);

    TIMERSTART(malloc_devices)
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu));
        cudaMalloc(&ying[gpu], sizeof(data_t)*batch_size_secure);
        cudaMalloc(&yang[gpu], sizeof(data_t)*batch_size_secure);
    } CUERR
    context.sync_hard();
    TIMERSTOP(malloc_devices)

    context.sync_all_streams();

    TIMERSTART(zero_gpu_buffers)
    const data_t init_data = 0;
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu){
        cudaSetDevice(context.get_device_id(gpu));
        memset_kernel<<<256, 1024, 0,
            context.get_streams(gpu)[0 % num_gpus]>>>
            (ying[gpu], batch_size_secure, init_data);
        memset_kernel<<<256, 1024, 0,
            context.get_streams(gpu)[1 % num_gpus]>>>
            (yang[gpu], batch_size_secure, init_data);
    }
    context.sync_all_streams();
    CUERR
    TIMERSTOP(zero_gpu_buffers)

    std::cout << "INFO: " << sizeof(data_t)*batch_size << " bytes" << std::endl;

    std::vector<data_t *> srcs(num_gpus);
    std::vector<data_t *> dsts(num_gpus);
    std::vector<size_t  > lens(num_gpus);

    // generate batch of data on main device
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs[gpu] = ying[gpu];
        dsts[gpu] = yang[gpu];
    }
    lens[main_gpu] = batch_size;

    TIMERSTART(init_data)
    cudaSetDevice(context.get_device_id(main_gpu));
    generate_data<<<256, 1024, 0, context.get_streams(main_gpu)[0]>>>
        (srcs[main_gpu], lens[main_gpu], size_t(0));
    context.sync_all_streams();
    CUERR
    TIMERSTOP(init_data)

    // perform multisplit on main device
    auto part_hash = [=] HOSTDEVICEQUALIFIER (const data_t& x){
        return (x % num_gpus);
    };

    std::vector<std::vector<size_t>> table(num_gpus, std::vector<size_t>(num_gpus));
    TIMERSTART(multisplit)
    multisplit.execAsync(srcs, lens, dsts, lens, table, part_hash);
    multisplit.sync();
    TIMERSTOP(multisplit)

    std::cout << "\nPartition Table:" << std::endl;
    for (gpu_id_t src = 0; src < num_gpus; src++)
        for (gpu_id_t trg = 0; trg < num_gpus; trg++)
            std::cout << table[src][trg] << (trg+1 == num_gpus ? "\n" : " ");
    std::cout << std::endl;

    // reset srcs to zero
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu));
        cudaMemsetAsync(srcs[gpu], 0, sizeof(data_t)*lens[gpu],
                        context.get_streams(gpu)[0]);
    } CUERR

    // perform broadcast
    std::vector<size_t> mems_lens(num_gpus);
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        srcs[gpu] = yang[gpu];
        dsts[gpu] = ying[gpu];
        mems_lens[gpu] = batch_size_secure;
    }

    // broadcast.show_plan();

    size_t total = 0;
    for(auto& t : table[main_gpu])
        total += t;

    TIMERSTART(broadcast)
    broadcast.execAsync(srcs[main_gpu], batch_size_secure, total, dsts, mems_lens);
    broadcast.sync();
    TIMERSTOP(broadcast)

    std::vector<size_t> prefix(num_gpus+1);
    for (gpu_id_t part = 0; part < num_gpus; part++) {
        prefix[part+1] = prefix[part] + table[main_gpu][part];
    }

    TIMERSTART(validate_broadcast)
    for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
        cudaSetDevice(context.get_device_id(gpu));
        for (gpu_id_t part = 0; part < num_gpus; part++)
            validate<<<256, 1024, 0, context.get_streams(gpu)[part]>>>
                (dsts[gpu]+prefix[part], table[main_gpu][part], part, part_hash);
    }
    context.sync_hard();
    CUERR
    TIMERSTOP(validate_broadcast)

    std::cout << '\n';

    context.sync_hard();
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu)); CUERR
        cudaFree(ying[gpu]);        CUERR
        cudaFree(yang[gpu]);        CUERR
    } CUERR
}

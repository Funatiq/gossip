#pragma once

#include <iostream>
#include <cstdint>
#include "include/gossip.cuh"
#include "include/hpc_helpers/include/cuda_helpers.cuh"
#include "include/hpc_helpers/include/timers.cuh"

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

template<
    typename data_t>
void memset_all(
    gossip::context_t& context,
    std::vector<data_t *>& data,
    const std::vector<size_t>& lengths,
    const data_t init_data = 0
) {
    gpu_id_t num_gpus = context.get_num_devices();
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu){
        cudaSetDevice(context.get_device_id(gpu));
        memset_kernel<<<256, 1024, 0, context.get_streams(gpu)[0]>>>
            (data[gpu], lengths[gpu], init_data);
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

template<
    typename data_t>
void generate_all(
    gossip::context_t& context,
    std::vector<data_t *>& data,
    const std::vector<size_t>& lengths
) {
    gpu_id_t num_gpus = context.get_num_devices();
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu){
        cudaSetDevice(context.get_device_id(gpu));
        generate_data<<<256, 1024, 0, context.get_streams(gpu)[0]>>>
            (data[gpu], lengths[gpu], gpu*lengths[gpu]);
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

void print_partition_table(const std::vector<std::vector<size_t>>& table) {
    std::cout << "\nPartition Table:" << std::endl;
    for (gpu_id_t src = 0; src < table.size(); src++) {
        for (gpu_id_t trg = 0; trg < table[src].size(); trg++)
            std::cout << table[src][trg] << ' ';
        std::cout << '\n';
    }
    std::cout << std::endl;
}

void print_buffer_sizes(const std::vector<size_t>& bufs_lens) {
    std::cout << "Required buffer sizes:" << std::endl;
    for (const auto& buf_len : bufs_lens) {
        std::cout << buf_len << ' ';
    }
    std::cout << '\n' << std::endl;
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
    const gpu_id_t num_gpus = context.get_num_devices();
    std::cout << "INFO: " << sizeof(data_t)*batch_size*num_gpus << " bytes (all2all)" << std::endl;

    std::vector<data_t *> srcs(num_gpus);
    std::vector<data_t *> dsts(num_gpus);
    const std::vector<size_t> lens(num_gpus, batch_size);
    const std::vector<size_t> mems_lens(num_gpus, batch_size_secure);

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "malloc_devices", context.get_device_id(0), std::cout);
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context.get_device_id(gpu)); CUERR
            cudaMalloc(&srcs[gpu], sizeof(data_t)*mems_lens[gpu]); CUERR
            cudaMalloc(&dsts[gpu], sizeof(data_t)*mems_lens[gpu]); CUERR
        }
    }

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "zero_gpu_buffers", context.get_device_id(0), std::cout);
        memset_all(context, srcs, mems_lens, data_t(0));
        memset_all(context, dsts, mems_lens, data_t(0));
        context.sync_all_streams();
        CUERR
    }

    // generate batch of data on each device
    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "init_data", context.get_device_id(0), std::cout);
        generate_all(context, srcs, lens);
        context.sync_all_streams();
        CUERR
    }

    // perform multisplit on each device
    auto part_hash = [=] DEVICEQUALIFIER (const data_t& x){
        return (x % num_gpus);
    };

    std::vector<std::vector<size_t>> table(num_gpus, std::vector<size_t>(num_gpus));
    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "multisplit", context.get_device_id(0), std::cout);
        multisplit.execAsync(srcs, lens, dsts, lens, table, part_hash);
        multisplit.sync();
    }

    print_partition_table(table);

    // prepare all2all --------------------------------------------------------
    srcs.swap(dsts);
    // reset dsts and buffer to zero
    memset_all(context, dsts, mems_lens, data_t(0));
    context.sync_all_streams();
    CUERR

    // all2all.show_plan();

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "all2all", context.get_device_id(0), std::cout);
        all2all.execAsync(srcs, mems_lens, dsts, mems_lens, table);
        all2all.sync();
    }

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "validate", context.get_device_id(0), std::cout);
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
    }

    // cleanup ----------------------------------------------------------------
    context.sync_hard();
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu)); CUERR
        cudaFree(srcs[gpu]);        CUERR
        cudaFree(dsts[gpu]);        CUERR
    }
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
    const gpu_id_t num_gpus = context.get_num_devices();
    std::cout << "INFO: " << sizeof(data_t)*batch_size*num_gpus << " bytes (all2all_async)" << std::endl;

    std::vector<data_t *> srcs(num_gpus);
    std::vector<data_t *> dsts(num_gpus);
    const std::vector<size_t> lens(num_gpus, batch_size);
    const std::vector<size_t> mems_lens(num_gpus, batch_size_secure);

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "malloc_devices", context.get_device_id(0), std::cout);
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context.get_device_id(gpu)); CUERR
            cudaMalloc(&srcs[gpu], sizeof(data_t)*mems_lens[gpu]); CUERR
            cudaMalloc(&dsts[gpu], sizeof(data_t)*mems_lens[gpu]); CUERR
        }
    }

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "zero_gpu_buffers", context.get_device_id(0), std::cout);
        memset_all(context, srcs, mems_lens, data_t(0));
        memset_all(context, dsts, mems_lens, data_t(0));
        context.sync_all_streams();
        CUERR
    }

    // generate batch of data on each device
    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "init_data", context.get_device_id(0), std::cout);
        generate_all(context, srcs, lens);
        context.sync_all_streams();
        CUERR
    }

    // perform multisplit on each device
    auto part_hash = [=] DEVICEQUALIFIER (const data_t& x){
        return (x % num_gpus);
    };

    std::vector<std::vector<size_t>> table(num_gpus, std::vector<size_t>(num_gpus));
    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "multisplit", context.get_device_id(0), std::cout);
        multisplit.execAsync(srcs, lens, dsts, lens, table, part_hash);
        multisplit.sync();
    }

    print_partition_table(table);

    // prepare all2all --------------------------------------------------------
    srcs.swap(dsts);

    std::vector<size_t> bufs_lens = all2all.calcBufferLengths(table);
    print_buffer_sizes(bufs_lens);

    std::vector<data_t *> bufs(num_gpus);
    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "malloc_buffers", context.get_device_id(0), std::cout);
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context.get_device_id(gpu)); CUERR
            cudaMalloc(&bufs[gpu], sizeof(data_t)*bufs_lens[gpu]); CUERR
        }
    }

    // reset dsts and buffer to zero
    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "reset_buffers", context.get_device_id(0), std::cout);
        memset_all(context, dsts, mems_lens, data_t(0));
        memset_all(context, bufs, bufs_lens, data_t(0));
        context.sync_all_streams();
        CUERR
    }

    // all2all.show_plan();

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "all2all_async", context.get_device_id(0), std::cout);
        all2all.execAsync(srcs, mems_lens, dsts, mems_lens, bufs, bufs_lens, table);
        all2all.sync();
    }

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "validate", context.get_device_id(0), std::cout);
        std::vector<size_t> lengths(num_gpus, 0);
        for (gpu_id_t trg = 0; trg < num_gpus; trg++) {
            for (gpu_id_t src = 0; src < num_gpus; src++)
                lengths[trg] += table[src][trg];
        }

        for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
            cudaSetDevice(context.get_device_id(gpu));
            validate<<<256, 1024, 0, context.get_streams(gpu)[0]>>>
                (dsts[gpu], lengths[gpu], gpu, part_hash);
        }
        CUERR
    }

    // cleanup ----------------------------------------------------------------
    context.sync_hard();
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu)); CUERR
        cudaFree(srcs[gpu]);        CUERR
        cudaFree(dsts[gpu]);        CUERR
        cudaFree(bufs[gpu]);        CUERR
    }
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
    const gpu_id_t num_gpus = context.get_num_devices();
    std::cout << "INFO: " << sizeof(data_t)*batch_size << " bytes (scatter_gather)" << std::endl;

    std::vector<data_t *> srcs(num_gpus);
    std::vector<data_t *> dsts(num_gpus);
    std::vector<size_t  > lens(num_gpus, 0);
    lens[main_gpu] = batch_size;
    const std::vector<size_t> mems_lens(num_gpus, batch_size_secure);

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "malloc_devices", context.get_device_id(0), std::cout);
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context.get_device_id(gpu)); CUERR
            cudaMalloc(&srcs[gpu], sizeof(data_t)*mems_lens[gpu]); CUERR
            cudaMalloc(&dsts[gpu], sizeof(data_t)*mems_lens[gpu]); CUERR
        }
    }

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "zero_gpu_buffers", context.get_device_id(0), std::cout);
        memset_all(context, srcs, mems_lens, data_t(0));
        memset_all(context, dsts, mems_lens, data_t(0));
        context.sync_all_streams();
        CUERR
    }

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "init_data", context.get_device_id(0), std::cout);
        cudaSetDevice(context.get_device_id(main_gpu));
        generate_data<<<256, 1024, 0, context.get_streams(main_gpu)[0]>>>
            (srcs[main_gpu], lens[main_gpu], size_t(0));
        context.sync_all_streams();
        CUERR
    }

    // perform multisplit on main device
    auto part_hash = [=] HOSTDEVICEQUALIFIER (const data_t& x){
        return (x % num_gpus);
    };

    std::vector<std::vector<size_t>> table(num_gpus, std::vector<size_t>(num_gpus));
    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "multisplit", context.get_device_id(0), std::cout);
        multisplit.execAsync(srcs, lens, dsts, lens, table, part_hash);
        multisplit.sync();
    }

    print_partition_table(table);

    // prepare scatter --------------------------------------------------------
    srcs.swap(dsts);

    std::vector<size_t> bufs_lens_scatter = scatter.calcBufferLengths(table[main_gpu]);
    print_buffer_sizes(bufs_lens_scatter);

    std::vector<data_t *> bufs(num_gpus);
    std::vector<size_t> bufs_lens(bufs_lens_scatter);
    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "malloc_buffers", context.get_device_id(0), std::cout);
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context.get_device_id(gpu)); CUERR
            cudaMalloc(&bufs[gpu], sizeof(data_t)*bufs_lens[gpu]); CUERR
        }
    }

    // reset dsts and buffer to zero
    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "reset_buffers", context.get_device_id(0), std::cout);
        memset_all(context, dsts, mems_lens, data_t(0));
        memset_all(context, bufs, bufs_lens, data_t(0));
        context.sync_all_streams();
        CUERR
    }

    // scatter.show_plan();

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "scatter", context.get_device_id(0), std::cout);
        scatter.execAsync(srcs[main_gpu], mems_lens[main_gpu],
                        dsts, mems_lens,
                        bufs, bufs_lens,
                        table[main_gpu]);
        scatter.sync();
        CUERR
    }

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "validate_scatter", context.get_device_id(0), std::cout);
        for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
            cudaSetDevice(context.get_device_id(gpu));
            validate<<<256, 1024, 0, context.get_streams(gpu)[0]>>>
                (dsts[gpu], table[main_gpu][gpu], gpu, part_hash);
        }
        context.sync_all_streams();
        CUERR
    }

    std::cout << '\n';

    // prepare gather ---------------------------------------------------------
    srcs.swap(dsts);

    std::vector<size_t> bufs_lens_gather = gather.calcBufferLengths(table[main_gpu]);
    print_buffer_sizes(bufs_lens_gather);

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "realloc_buffers", context.get_device_id(0), std::cout);
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            if(bufs_lens[gpu] < bufs_lens_gather[gpu]) {
                bufs_lens[gpu] = bufs_lens_gather[gpu];
                cudaSetDevice(context.get_device_id(gpu)); CUERR
                cudaFree(bufs[gpu]); CUERR
                cudaMalloc(&bufs[gpu], sizeof(data_t)*bufs_lens[gpu]); CUERR
            }
        }
    }

    // reset dsts and buffer to zero
    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "reset_buffers_again", context.get_device_id(0), std::cout);
        memset_all(context, dsts, mems_lens, data_t(0));
        memset_all(context, bufs, bufs_lens, data_t(0));
        context.sync_all_streams();
        CUERR
    }

    // gather.show_plan();

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "gather", context.get_device_id(0), std::cout);
        gather.execAsync(srcs, mems_lens,
                        dsts[main_gpu], mems_lens[main_gpu],
                        bufs, bufs_lens,
                        table[main_gpu]);
        gather.sync();
        CUERR
    }

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "validate_gather", context.get_device_id(0), std::cout);
        std::vector<data_t *> mems2(num_gpus, dsts[main_gpu]);
        for (gpu_id_t trg = 1; trg < num_gpus; trg++) {
            mems2[trg] = mems2[trg-1] + table[main_gpu][trg-1];
        }

        cudaSetDevice(context.get_device_id(main_gpu));
        for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
            validate<<<256, 1024, 0, context.get_streams(main_gpu)[0]>>>
                (mems2[gpu], table[main_gpu][gpu], gpu, part_hash);
        }
        context.sync_all_streams();
        CUERR
    }

    // cleanup ----------------------------------------------------------------
    context.sync_hard();
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu)); CUERR
        cudaFree(srcs[gpu]);        CUERR
        cudaFree(dsts[gpu]);        CUERR
        cudaFree(bufs[gpu]);        CUERR
    }
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
    const gpu_id_t num_gpus = context.get_num_devices();
    const gpu_id_t main_gpu = 0;
    std::cout << "INFO: " << sizeof(data_t)*batch_size << " bytes (broadcast)" << std::endl;

    std::vector<data_t *> srcs(num_gpus);
    std::vector<data_t *> dsts(num_gpus);
    std::vector<size_t  > lens(num_gpus, 0);
    lens[main_gpu] = batch_size;
    const std::vector<size_t> mems_lens(num_gpus, batch_size_secure);

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "malloc_devices", context.get_device_id(0), std::cout);
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context.get_device_id(gpu)); CUERR
            cudaMalloc(&srcs[gpu], sizeof(data_t)*mems_lens[gpu]); CUERR
            cudaMalloc(&dsts[gpu], sizeof(data_t)*mems_lens[gpu]); CUERR
        }
    }

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "zero_gpu_buffers", context.get_device_id(0), std::cout);
        memset_all(context, srcs, mems_lens, data_t(0));
        memset_all(context, dsts, mems_lens, data_t(0));
        context.sync_all_streams();
        CUERR
    }

    // generate batch of data on main device
    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "init_data", context.get_device_id(0), std::cout);
        cudaSetDevice(context.get_device_id(main_gpu));
        generate_data<<<256, 1024, 0, context.get_streams(main_gpu)[0]>>>
            (srcs[main_gpu], lens[main_gpu], size_t(0));
        context.sync_all_streams();
        CUERR
    }

    // perform multisplit on main device
    auto part_hash = [=] HOSTDEVICEQUALIFIER (const data_t& x){
        return (x % num_gpus);
    };

    std::vector<std::vector<size_t>> table(num_gpus, std::vector<size_t>(num_gpus));
    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "multisplit", context.get_device_id(0), std::cout);
        multisplit.execAsync(srcs, lens, dsts, lens, table, part_hash);
        multisplit.sync();
    }

    print_partition_table(table);

    // prepare broadcast ------------------------------------------------------
    srcs.swap(dsts);
    // reset dsts to zero
    memset_all(context, dsts, mems_lens, data_t(0));
    context.sync_all_streams();
    CUERR

    // broadcast.show_plan();

    size_t total = 0;
    for(auto& t : table[main_gpu])
        total += t;

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "broadcast", context.get_device_id(0), std::cout);
        broadcast.execAsync(srcs[main_gpu], mems_lens[main_gpu], total, dsts, mems_lens);
        broadcast.sync();
    }

    std::vector<size_t> prefix(num_gpus+1);
    for (gpu_id_t part = 0; part < num_gpus; part++) {
        prefix[part+1] = prefix[part] + table[main_gpu][part];
    }

    {
        helpers::GpuTimer gtimer(context.get_streams(0)[0], "validate_broadcast", context.get_device_id(0), std::cout);
        for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
            cudaSetDevice(context.get_device_id(gpu));
            for (gpu_id_t part = 0; part < num_gpus; part++)
                validate<<<256, 1024, 0, context.get_streams(gpu)[part]>>>
                    (dsts[gpu]+prefix[part], table[main_gpu][part], part, part_hash);
        }
        context.sync_all_streams();
        CUERR
    }

    std::cout << '\n';

    // cleanup ----------------------------------------------------------------
    context.sync_hard();
    for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(context.get_device_id(gpu)); CUERR
        cudaFree(srcs[gpu]);        CUERR
        cudaFree(dsts[gpu]);        CUERR
    }
}

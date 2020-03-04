#pragma once

#include "config.h"
#include "error_checking.hpp"
#include "context.cuh"

namespace gossip {

template <
    typename value_t,
    typename index_t,
    typename cnter_t,
    typename funct_t,
    typename desir_t> __global__
void binary_split(
    value_t * src,
    value_t * dst,
    index_t   len,
    cnter_t * counter,
    funct_t   part_hash,
    desir_t   desired
) {
    const auto thid = blockDim.x*blockIdx.x + threadIdx.x;

    for(index_t i = thid; i < len; i += gridDim.x*blockDim.x) {

        const value_t value = src[i];

        if (part_hash(value) == desired) {
            const index_t j = atomicAggInc(counter);
            dst[j] = value;
        }
    }
}

class multisplit_t {

    const context_t * context;
    std::vector<cnter_t *> counters_device;
    std::vector<cnter_t *> counters_host;

public:
    multisplit_t (const context_t& context_) : context(&context_)
    {
        check(context->is_valid(),
              "You have to pass a valid context!");

        initialize();
    }

private:
    void initialize() {
        counters_device.resize(get_num_devices());
        counters_host.resize(get_num_devices());

        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            cudaMalloc(&counters_device[gpu], sizeof(cnter_t));
            cudaMallocHost(&counters_host[gpu], sizeof(cnter_t)*get_num_devices());
        } CUERR
    }


public:
    ~multisplit_t () {

        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            cudaFreeHost(counters_host[gpu]);
            cudaFree(counters_device[gpu]);
        } CUERR
    }

    template <
        typename value_t,
        typename index_t,
        typename table_t,
        typename funct_t>
    bool execAsync (
        const std::vector<value_t *>& srcs,
        const std::vector<index_t  >& srcs_lens,
        const std::vector<value_t *>& dsts,
        const std::vector<index_t  >& dsts_lens,
        std::vector<std::vector<table_t> >& partition_table,
        funct_t functor
    ) const {

        if (!check(srcs.size() == get_num_devices(),
                    "srcs size does not match number of gpus."))
            return false;
        if (!check(srcs_lens.size() == get_num_devices(),
                    "srcs_lens size does not match number of gpus."))
            return false;
        if (!check(dsts.size() == get_num_devices(),
                    "dsts size does not match number of gpus."))
            return false;
        if (!check(dsts_lens.size() == get_num_devices(),
                    "dsts_lens size does not match number of gpus."))
            return false;
        if (!check(partition_table.size() == get_num_devices(),
                    "table size does not match number of gpus."))
            return false;
        for (const auto& counts : partition_table) {
            if (!check(counts.size() == get_num_devices(),
                        "table size does not match number of gpus."))
                return false;
        }

        for (gpu_id_t src_gpu = 0; src_gpu < get_num_devices(); ++src_gpu) {
            if (!check(srcs_lens[src_gpu] <= dsts_lens[src_gpu],
                        "dsts_lens too small for given srcs_lens."))
                return false;
        }

        // initialize the counting atomics with zeroes
        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            cudaMemsetAsync(counters_device[gpu], 0, sizeof(cnter_t),
                            context->get_streams(gpu)[0]);
        } CUERR

        // perform warp aggregated compression for each GPU independently
        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            for (gpu_id_t part = 0; part < get_num_devices(); ++part) {

                binary_split<<<256, 1024, 0, context->get_streams(gpu)[0]>>>
                   (srcs[gpu], dsts[gpu], srcs_lens[gpu],
                    counters_device[gpu], functor, part);
                cudaMemcpyAsync(&counters_host[gpu][part],
                                &counters_device[gpu][0],
                                sizeof(cnter_t), cudaMemcpyDeviceToHost,
                                context->get_streams(gpu)[0]);

            }
        } CUERR

        // this sync is mandatory
        sync();

        // recover the partition table from accumulated counters
        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu)
            for (gpu_id_t part = 0; part < get_num_devices(); ++part)
                partition_table[gpu][part] = (part == 0) ?
                                             counters_host[gpu][part] :
                                             counters_host[gpu][part] -
                                             counters_host[gpu][part-1];

        return true;
    }

    gpu_id_t get_num_devices () const noexcept {
        return context->get_num_devices();
    }

    void sync () const noexcept {
        context->sync_all_streams();
    }

    void sync_hard () const noexcept {
        context->sync_hard();
    }

    const context_t& get_context() const noexcept {
        return *context;
    }
};

} // namespace

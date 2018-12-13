#pragma once

#include "config.h"
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

template <
    bool throw_exceptions=true,
    typename cnter_t=uint32_t>
class multisplit_t {

    const context_t<> * context;
    bool external_context;
    std::vector<cnter_t *> counters_device;
    std::vector<cnter_t *> counters_host;

public:

    multisplit_t (
        const gpu_id_t num_gpus_)
        : external_context (false)
    {
        context = new context_t<>(num_gpus_);

        initialize();
    }

    multisplit_t (
        const std::vector<gpu_id_t>& device_ids_)
        : external_context (false)
    {
        context = new context_t<>(device_ids_);

        initialize();
    }

    multisplit_t (
        context_t<> * context_)
    : context(context_),
      external_context (true) {
        if (throw_exceptions)
            if (!context->is_valid())
                throw std::invalid_argument(
                    "You have to pass a valid context!"
                );

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

        if (!external_context)
            delete context;
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
        std::vector<std::vector<table_t> >& table,
        funct_t functor) const {

        if (srcs.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "srcs size does not match number of gpus.");
            else return false;
        if (srcs_lens.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "srcs_lens size does not match number of gpus.");
            else return false;
        if (dsts.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "dsts size does not match number of gpus.");
            else return false;
        if (dsts_lens.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "dsts_lens size does not match number of gpus.");
            else return false;
        if (table.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "table size does not match number of gpus.");
            else return false;
        for (const auto& t : table)
            if (t.size() != get_num_devices())
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "table size does not match number of gpus.");
                else return false;

        for (gpu_id_t src_gpu = 0; src_gpu < get_num_devices(); ++src_gpu) {
            if (srcs_lens[src_gpu] > dsts_lens[src_gpu])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "dsts_lens too small for given srcs_lens."
                    );
                else return false;
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
                    counters_device[gpu], functor, part+1);
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
                table[gpu][part] = part == 0 ? counters_host[gpu][part] :
                                   counters_host[gpu][part] -
                                   counters_host[gpu][part-1];

        // reset srcs to zero
        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            cudaMemsetAsync(srcs[gpu], 0, sizeof(value_t)*srcs_lens[gpu],
                            context->get_streams(gpu)[0]);
        } CUERR

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
};

} // namespace

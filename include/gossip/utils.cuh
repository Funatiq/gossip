#pragma once

#include "error_checking.hpp"
#include "context.cuh"

namespace gossip {

    // only for convenience
    template <
        typename value_t,
        typename index_t>
    bool clear(
        const context_t * context,
        const std::vector<value_t *>& mem,
        const std::vector<index_t  >& mem_lens
    ) {
        if(!check(mem.size() == context->get_num_devices(),
                 "mem size does not match number of gpus."))
            return false;
        if(!check(mem_lens.size() == context->get_num_devices(),
                 "mem_lens size does not match number of gpus."))
            return false;

        context->sync_all_streams();
        for (gpu_id_t gpu = 0; gpu < context->get_num_devices(); gpu++) {
            const gpu_id_t id = context->get_device_id(gpu);
            const auto stream = context->get_streams(gpu)[0];
            cudaSetDevice(id);
            const size_t size = mem_lens[gpu]
                              * sizeof(value_t);
            cudaMemsetAsync(mem[gpu], 0, size, stream);
        } CUERR

        return true;
    }

} // namespace

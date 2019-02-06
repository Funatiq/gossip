#pragma once

#include "config.h"
#include "context.cuh"

namespace gossip {

class point2point_t {

    const context_t * context;
    bool external_context;

public:

    point2point_t (
        const gpu_id_t num_gpus_)
        : context( new context_t(num_gpus_) ),
          external_context (false)
    {}

    point2point_t (
        const std::vector<gpu_id_t>& device_ids_)
        : context( new context_t(device_ids_) ),
          external_context (false)
    {}

    point2point_t (
        const context_t * context_)
        : context(context_),
          external_context (true)
    {
        check(context->is_valid(),
              "You have to pass a valid context!");
    }

    ~point2point_t () {
        if (!external_context)
            delete context;
    }

    template <
        cudaMemcpyKind cudaMemcpyDirection,
        typename value_t,
        typename index_t>
    bool execAsync (
        const std::vector<value_t *>& srcs,
        const std::vector<value_t *>& dsts,
        const std::vector<index_t  >& lens
    ) const {
        if (!check(srcs.size() == get_num_devices(),
                    "srcs size does not match number of gpus."))
            return false;
        if (!check(dsts.size() == get_num_devices(),
                    "dsts size does not match number of gpus."))
            return false;
        if (!check(lens.size() == get_num_devices(),
                    "lens size does not match number of gpus."))
            return false;

        for (gpu_id_t src_gpu = 0; src_gpu < get_num_devices(); ++src_gpu) {
            if (lens[src_gpu] > 0) {
                cudaSetDevice(context->get_device_id(src_gpu));
                cudaMemcpyAsync(dsts[src_gpu], srcs[src_gpu],
                                sizeof(value_t)*lens[src_gpu],
                                cudaMemcpyDirection,
                                context->get_streams(src_gpu)[0]);
            }
        } CUERR

        return true;
    }

    template <
        typename value_t,
        typename index_t>
    bool execH2DAsync (
        const std::vector<value_t *>& srcs,
        const std::vector<value_t *>& dsts,
        const std::vector<index_t  >& lens
    ) const {
        return execAsync<cudaMemcpyHostToDevice>(srcs, dsts, lens);
    }

    template <
        typename value_t,
        typename index_t>
    bool execD2HAsync (
        const std::vector<value_t *>& srcs,
        const std::vector<value_t *>& dsts,
        const std::vector<index_t  >& lens
    ) const {
        return execAsync<cudaMemcpyDeviceToHost>(srcs, dsts, lens);
    }

    template <
        typename value_t,
        typename index_t>
    bool execD2DAsync (
        const std::vector<value_t *>& srcs,
        const std::vector<value_t *>& dsts,
        const std::vector<index_t  >& lens
    ) const {
        return execAsync<cudaMemcpyDeviceToDevice>(srcs, dsts, lens);
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

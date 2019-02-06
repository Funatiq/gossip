#pragma once

#include <cstring>

#include "config.h"
#include "error_checking.hpp"
#include "context.cuh"

namespace gossip {

class memory_manager_t {

    const context_t * context;

public:
    memory_manager_t (const context_t& context_) : context(&context_)
    {
        check(context->is_valid(),
              "You have to pass a valid context!");
    }

    template <
        typename value_t,
        typename index_t>
    std::vector<value_t *>
    alloc_device(
        const std::vector<index_t>& lens,
        const bool zero=true) const {

        std::vector<value_t *> data = {};

        if (!check(lens.size() == get_num_devices(),
                    "lens size does not match number of gpus."))
            return data;

        data.resize(get_num_devices());

        // malloc as device-sided memory
        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            cudaMalloc(&data[gpu], sizeof(value_t)*lens[gpu]);
            if (zero)
                cudaMemsetAsync(data[gpu], 0, sizeof(value_t)*lens[gpu],
                                context->get_streams(gpu)[0]);
        }
        CUERR

        return std::move(data);
    }

    template <
        typename value_t,
        typename index_t>
    std::vector<value_t *>
    alloc_host(
        const std::vector<index_t>& lens,
        const bool zero=true) const {

        std::vector<value_t *> data = {};

        if (!check(lens.size() == get_num_devices(),
                    "lens size does not match number of gpus."))
            return data;

        data.resize(get_num_devices());

        // malloc as host-sided pinned memory
        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu) {
            cudaMallocHost(&data[gpu], sizeof(value_t)*lens[gpu]);
            if (zero)
                std::memset(data[gpu], 0, sizeof(value_t)*lens[gpu]);
        }
        CUERR

        return std::move(data);
    }

    template <
        typename value_t>
    bool free_device(std::vector<value_t *>& data) const {

        if (!check(data.size() == get_num_devices(),
                    "data size does not match number of gpus."))
            return false;

        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            cudaFree(data[gpu]);
        }
        CUERR

        return true;
    }

    template <
        typename value_t>
    bool free_host(std::vector<value_t *>& data) const {

        if (!check(data.size() == get_num_devices(),
                    "data size does not match number of gpus."))
            return false;

        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu)
            cudaFreeHost(data[gpu]);
        CUERR

        return true;
    }

    gpu_id_t get_num_devices () const noexcept {
        return context->get_num_devices();
    }
};

} // namespace

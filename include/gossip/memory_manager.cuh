#pragma once

template<
    bool throw_exceptions=true>
class memory_manager_t {

    gpu_id_t num_gpus;
    context_t<> * context;
    bool external_context;

public:

    memory_manager_t (
        const gpu_id_t num_gpus_)
        : external_context (false) {

        context = new context_t<>(num_gpus_);
        num_gpus = context->get_num_devices();
    }

    memory_manager_t (
        const std::vector<gpu_id_t>& device_ids_)
        : external_context (false) {

        context = new context_t<>(device_ids_);
        num_gpus = context->get_num_devices();
    }

    memory_manager_t (
        context_t<> * context_) : context(context_),
                                  external_context (true) {
            if (throw_exceptions)
                if (!context->is_valid())
                    throw std::invalid_argument(
                        "You have to pass a valid context!"
                    );

        num_gpus = context->get_num_devices();
    }

    ~memory_manager_t () {
        if (!external_context)
            delete context;
    }


    template <
        typename value_t,
        typename index_t>
    std::vector<value_t *>
    alloc_device(
        const std::vector<index_t>& lens,
        const bool zero=true) const {

        std::vector<value_t *> data = {};

        if (lens.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "lens size does not match number of gpus.");
            else return data;

        data.resize(num_gpus);

        // malloc as device-sided memory
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
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

        if (lens.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "lens size does not match number of gpus.");
            else return data;

        data.resize(num_gpus);

        // malloc as host-sided pinned memory
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
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

        if (data.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "data size does not match number of gpus.");
            else return false;

        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            cudaFree(data[gpu]);
        }
        CUERR

        return true;
    }

    template <
        typename value_t>
    bool free_host(std::vector<value_t *>& data) const {

        if (data.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "data size does not match number of gpus.");
            else return false;

        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu)
            cudaFreeHost(data[gpu]);
        CUERR

        return true;
    }
};

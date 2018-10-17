#pragma once

template<
    gpu_id_t num_gpus,
    bool throw_exceptions=true>
class memory_manager_t {

    context_t<num_gpus> * context;
    bool external_context;

public:

    memory_manager_t (
        std::vector<gpu_id_t>& device_ids_ = std::vector<gpu_id_t>{})
        : external_context (false) {

        context = new context_t<num_gpus>(device_ids_);
    }

    memory_manager_t (
        context_t<num_gpus> * context_) : context(context_),
                                          external_context (true) {
            if (throw_exceptions)
                if (!context->is_valid())
                    throw std::invalid_argument(
                        "You have to pass a valid context!"
                    );
    }

    ~memory_manager_t () {
        if (!external_context)
            delete context;
    }


    template <
        typename value_t,
        typename index_t>
    std::array<value_t *, num_gpus>
    alloc_device(
        const std::array<index_t, num_gpus>& lens,
        const bool zero=true) const {

        std::array<value_t *, num_gpus> data;

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
    std::array<value_t *, num_gpus>
    alloc_host(
        const std::array<index_t, num_gpus>& lens,
        const bool zero=true) const {

        std::array<value_t *, num_gpus> data;

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
    void free_device(std::array<value_t *, num_gpus>& data) const {

        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            cudaFree(data[gpu]);
        }
        CUERR
    }

    template <
        typename value_t>
    void free_host(std::array<value_t *, num_gpus>& data) const {

        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu)
            cudaFreeHost(data[gpu]);
        CUERR
    }
};

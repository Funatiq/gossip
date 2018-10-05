#pragma once

template<
    uint64_t num_gpus,
    uint64_t throw_exceptions=true>
class memory_manager_t {

    context_t<num_gpus> * context;
    bool external_context;

public:

    memory_manager_t (
        uint64_t * device_ids_=0) : external_context (false) {

        if (device_ids_)
            context = new context_t<num_gpus>(device_ids_);
        else
            context = new context_t<num_gpus>();
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
    value_t ** alloc_device(index_t lens[num_gpus], bool zero=true) const {

        value_t ** data = new value_t*[num_gpus];

        // malloc as device-sided memory
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            cudaMalloc(&data[gpu], sizeof(value_t)*lens[gpu]);
            if (zero)
                cudaMemsetAsync(data[gpu], 0, sizeof(value_t)*lens[gpu],
                                context->get_streams(gpu)[0]);
        }
        CUERR

        return data;
    }

    template <
        typename value_t,
        typename index_t>
    value_t ** alloc_host(index_t lens[num_gpus], bool zero=true) const {

        value_t ** data = new value_t*[num_gpus];

        // malloc as host-sided pinned memory
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaMallocHost(&data[gpu], sizeof(value_t)*lens[gpu]);
            if (zero)
                std::memset(data[gpu], 0, sizeof(value_t)*lens[gpu]);
        }
        CUERR

        return data;
    }

    template <
        typename value_t>
    void free_device(value_t ** data) const {

        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            cudaFree(data[gpu]);
        }
        CUERR

        delete [] data;
    }

    template <
        typename value_t>
    void free_host(value_t ** data) const {

        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
            cudaFreeHost(data[gpu]);
        CUERR

        delete [] data;
    }
};

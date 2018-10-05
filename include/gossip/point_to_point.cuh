#pragma once

template <
    uint64_t num_gpus,
    uint64_t throw_exceptions=true>
class point2point_t {

    const context_t<num_gpus> * context;
    bool external_context;

public:

    point2point_t (
        uint64_t * device_ids_=0) : external_context (false) {

        if (device_ids_)
            context = new context_t<num_gpus>(device_ids_);
        else
            context = new context_t<num_gpus>();
    }

    point2point_t (
        context_t<num_gpus> * context_) : context(context_),
                                          external_context (true) {
            if (throw_exceptions)
                if (!context->is_valid())
                    throw std::invalid_argument(
                        "You have to pass a valid context!"
                    );
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
        value_t * srcs[num_gpus],
        value_t * dsts[num_gpus],
        index_t   lens[num_gpus]) const noexcept {

        for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
            cudaSetDevice(context->get_device_id(src_gpu));
            cudaMemcpyAsync(dsts[src_gpu], srcs[src_gpu],
                            sizeof(value_t)*lens[src_gpu],
                            cudaMemcpyDirection,
                            context->get_streams(src_gpu)[0]);
        } CUERR

        return true;
    }

    template <
        typename value_t,
        typename index_t>
    bool execH2DAsync (
        value_t * srcs[num_gpus],
        value_t * dsts[num_gpus],
        index_t   lens[num_gpus]) const noexcept {

        return execAsync<cudaMemcpyHostToDevice>(srcs, dsts, lens);
    }

    template <
        typename value_t,
        typename index_t>
    bool execD2HAsync (
        value_t * srcs[num_gpus],
        value_t * dsts[num_gpus],
        index_t   lens[num_gpus]) const noexcept {

        return execAsync<cudaMemcpyDeviceToHost>(srcs, dsts, lens);
    }

    template <
        typename value_t,
        typename index_t>
    bool execD2DAsync (
        value_t * srcs[num_gpus],
        value_t * dsts[num_gpus],
        index_t   lens[num_gpus]) const noexcept {

        return execAsync<cudaMemcpyDeviceToDevice>(srcs, dsts, lens);
    }

    void sync () const noexcept {
        context->sync_all_streams();
    }

    void sync_hard () const noexcept {
        context->sync_hard();
    }
};

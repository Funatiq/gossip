#pragma once

template <
    bool throw_exceptions=true>
class point2point_t {

    gpu_id_t num_gpus;
    const context_t<> * context;
    bool external_context;

public:

    point2point_t (
        const gpu_id_t num_gpus_)
        : external_context (false) {

        context = new context_t<>(num_gpus_);
        num_gpus = context->get_num_devices();
    }

    point2point_t (
        const std::vector<gpu_id_t>& device_ids_)
        : external_context (false) {

        context = new context_t<>(device_ids_);
        num_gpus = context->get_num_devices();
    }

    point2point_t (
        context_t<> * context_) : context(context_),
                                  external_context (true) {
            if (throw_exceptions)
                if (!context->is_valid())
                    throw std::invalid_argument(
                        "You have to pass a valid context!"
                    );

        num_gpus = context->get_num_devices();
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
        const std::vector<index_t  >& lens) const {

        if (srcs.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "srcs size does not match number of gpus.");
            else return false;
        if (dsts.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "dsts size does not match number of gpus.");
            else return false;
        if (lens.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "lens size does not match number of gpus.");
            else return false;

        for (gpu_id_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
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
        const std::vector<value_t *>& srcs,
        const std::vector<value_t *>& dsts,
        const std::vector<index_t  >& lens) const {

        return execAsync<cudaMemcpyHostToDevice>(srcs, dsts, lens);
    }

    template <
        typename value_t,
        typename index_t>
    bool execD2HAsync (
        const std::vector<value_t *>& srcs,
        const std::vector<value_t *>& dsts,
        const std::vector<index_t  >& lens) const {

        return execAsync<cudaMemcpyDeviceToHost>(srcs, dsts, lens);
    }

    template <
        typename value_t,
        typename index_t>
    bool execD2DAsync (
        const std::vector<value_t *>& srcs,
        const std::vector<value_t *>& dsts,
        const std::vector<index_t  >& lens) const {

        return execAsync<cudaMemcpyDeviceToDevice>(srcs, dsts, lens);
    }

    void sync () const noexcept {
        context->sync_all_streams();
    }

    void sync_hard () const noexcept {
        context->sync_hard();
    }
};

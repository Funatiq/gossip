#pragma once

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
    desir_t   desired) {

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
    gpu_id_t num_gpus,
    bool throw_exceptions=true,
    typename cnter_t=uint32_t>
class multisplit_t {

    context_t<num_gpus> * context;
    bool external_context;
    std::array<cnter_t *, num_gpus> counters_device;
    std::array<cnter_t *, num_gpus> counters_host;

public:

    multisplit_t (
        std::vector<gpu_id_t>& device_ids_ = std::vector<gpu_id_t>{})
        : external_context (false) {

        context = new context_t<num_gpus>(device_ids_);

        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            cudaMalloc(&counters_device[gpu], sizeof(cnter_t));
            cudaMallocHost(&counters_host[gpu], sizeof(cnter_t)*num_gpus);
        } CUERR
    }

    multisplit_t (
        context_t<num_gpus> * context_) : context(context_),
                                          external_context (true) {
        if (throw_exceptions)
            if (!context->is_valid())
                throw std::invalid_argument(
                    "You have to pass a valid context!"
                );

        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context->get_device_id(gpu)); 
            cudaMalloc(&counters_device[gpu], sizeof(cnter_t));
            cudaMallocHost(&counters_host[gpu], sizeof(cnter_t)*num_gpus);
        } CUERR
    }

    ~multisplit_t () {

        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
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
        const std::array<value_t *, num_gpus>& srcs,
        const std::array<index_t  , num_gpus>& srcs_lens,
        const std::array<value_t *, num_gpus>& dsts,
        const std::array<index_t  , num_gpus>& dsts_lens,
        std::array<std::array<table_t, num_gpus>, num_gpus>& table,
        funct_t functor) const noexcept {

        for (gpu_id_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
            if (srcs_lens[src_gpu] > dsts_lens[src_gpu])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "dsts_lens too small for given srcs_lens."
                    );
                else return false;
        }

        // initialize the counting atomics with zeroes
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            cudaMemsetAsync(counters_device[gpu], 0, sizeof(cnter_t),
                            context->get_streams(gpu)[0]);
        } CUERR

        // perform warp aggregated compression for each GPU independently
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            for (gpu_id_t part = 0; part < num_gpus; ++part) {

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
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu)
            for (gpu_id_t part = 0; part < num_gpus; ++part)
                table[gpu][part] = part == 0 ? counters_host[gpu][part] :
                                   counters_host[gpu][part] -
                                   counters_host[gpu][part-1];

        // reset srcs to zero
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(context->get_device_id(gpu));
            cudaMemsetAsync(srcs[gpu], 0, sizeof(value_t)*srcs_lens[gpu],
                            context->get_streams(gpu)[0]);
        } CUERR

        return true;
    }

    void sync () const noexcept {
        context->sync_all_streams();
    }

    void sync_hard () const noexcept {
        context->sync_hard();
    }
};

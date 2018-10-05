#pragma once

template<
    uint64_t num_gpus,
    uint64_t throw_exceptions=true>
class all2all_t {

    context_t<num_gpus> * context;
    bool external_context;

public:

    all2all_t (
        uint64_t * device_ids_=0) : external_context (false){

        if (device_ids_)
            context = new context_t<num_gpus>(device_ids_);
        else
            context = new context_t<num_gpus>();
    }

    all2all_t (
        context_t<num_gpus> * context_) : context(context_),
                                          external_context (true) {
            if (throw_exceptions)
                if (!context->is_valid())
                    throw std::invalid_argument(
                        "You have to pass a valid context!"
                    );
    }

    ~all2all_t () {
        if (!external_context)
            delete context;
    }

    template <
        typename value_t,
        typename index_t,
        typename table_t>
    bool execAsync (
        value_t * srcs[num_gpus],        // src[k] resides on device_ids[k]
        index_t srcs_lens[num_gpus],     // src_len[k] is length of src[k]
        value_t * dsts[num_gpus],        // dst[k] resides on device_ids[k]
        index_t dsts_lens[num_gpus],     // dst_len[0] is length of dst[k]
        table_t table[num_gpus][num_gpus]) const {  // [src_gpu, partition]

        // syncs with zero stream in order to enforce sequential
        // consistency with traditional synchronous memcpy calls
        if (!external_context)
            context->sync_hard();

        // compute prefix sums over the partition table
        uint64_t h_table[num_gpus][num_gpus+1] = {0}; // horizontal scan
        uint64_t v_table[num_gpus+1][num_gpus] = {0}; // vertical scan

        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
            for (uint64_t part = 0; part < num_gpus; ++part) {
                h_table[gpu][part+1] = table[gpu][part]+h_table[gpu][part];
                v_table[gpu+1][part] = table[gpu][part]+v_table[gpu][part];
            }
        }

        // check src_lens for compatibility
        bool valid_srcs_lens = true;
        for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
            valid_srcs_lens &= h_table[src_gpu][num_gpus]
                            <= srcs_lens[src_gpu];
        if (!valid_srcs_lens)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "srcs_lens not compatible with partition_table.");
            else return false;

        // check dst_lens for compatibility
        bool valid_dsts_lens = true;
        for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu)
            valid_dsts_lens &= v_table[num_gpus][dst_gpu]
                            <= dsts_lens[dst_gpu];
        if (!valid_dsts_lens)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "dsts_lens not compatible with partition_table.");
            else return false;

        // issue asynchronous copies
        for (uint64_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
            const uint64_t src = context->get_device_id(src_gpu);
            cudaSetDevice(src);
            for (uint64_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                const uint64_t dst = context->get_device_id(dst_gpu);
                const uint64_t len = table[src_gpu][dst_gpu];
                value_t * from = srcs[src_gpu] + h_table[src_gpu][dst_gpu];
                value_t * to   = dsts[dst_gpu] + v_table[src_gpu][dst_gpu];

                cudaMemcpyPeerAsync(to, dst, from, src,
                                    len*sizeof(value_t),
                                    context->get_streams(src_gpu)[dst_gpu]);

            } CUERR
        }

        return true;
    }

    void print_connectivity_matrix () const noexcept {
        context->print_connectivity_matrix();
    }

    void sync () const noexcept {
        context->sync_all_streams();
    }

    void sync_hard () const noexcept {
        context->sync_hard();
    }
};

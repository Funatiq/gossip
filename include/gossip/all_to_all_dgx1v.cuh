#pragma once

template<
    gpu_id_t num_gpus,
    bool throw_exceptions=true>
class all2all_dgx1v_t {

    context_t<num_gpus> * context;
    bool external_context;

    static_assert(num_gpus==8, "currently only for exactly all GPUs.");

public:
    all2all_dgx1v_t (
        gpu_id_t * device_ids_=0) : external_context (false){

        if (device_ids_)
            context = new context_t<num_gpus>(device_ids_);
        else
            context = new context_t<num_gpus>();
    }

    all2all_dgx1v_t (
        context_t<num_gpus> * context_) : context(context_),
                                          external_context (true) {
            if (throw_exceptions)
                if (!context->is_valid())
                    throw std::invalid_argument(
                        "You have to pass a valid context!"
                    );
    }

    ~all2all_dgx1v_t () {
        if (!external_context)
            delete context;
    }

private:
    struct transfer {
        const gpu_id_t src_gpu;
        const size_t src_pos;
        const gpu_id_t trg_gpu;
        const size_t trg_pos;
        const size_t len;

        transfer(const gpu_id_t src_gpu,
                 const size_t src_pos,
                 const gpu_id_t trg_gpu,
                 const size_t trg_pos,
                 const size_t len) :
            src_gpu(src_gpu),
            src_pos(src_pos),
            trg_gpu(trg_gpu),
            trg_pos(trg_pos),
            len(len)
        {}
    };

    template<typename table_t>
    struct transfer_handler {
        std::vector<transfer> phase_one = {};
        std::vector<transfer> phase_two = {};

        size_t phase_one_offsets[num_gpus] = {0};
        size_t phase_two_offsets[num_gpus] = {0};

        const table_t (&table)[num_gpus][num_gpus];
        size_t h_table[num_gpus][num_gpus+1] = {{0}}; // horizontal scan

        transfer_handler(const table_t (&table)[num_gpus][num_gpus]) : table(table) {
            for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
                for (gpu_id_t part = 0; part < num_gpus; ++part) {
                    h_table[gpu][part+1] = table[gpu][part]+h_table[gpu][part];
                }
            }
        }

        void push_back(const gpu_id_t src, const gpu_id_t proxy, const gpu_id_t trg) {
            const size_t transfer_size = table[src][trg];
            phase_one.emplace_back(src, h_table[src][trg], proxy, phase_one_offsets[proxy], transfer_size);
            phase_two.emplace_back(proxy, phase_one_offsets[proxy], trg, phase_two_offsets[trg], transfer_size);
            phase_one_offsets[proxy] += transfer_size;
            phase_two_offsets[trg] += transfer_size;
        }

        void one_to_all(const gpu_id_t src, const std::array<gpu_id_t, num_gpus>& proxies) {
            for (gpu_id_t trg = 0; trg < num_gpus; ++trg) {
                push_back(src, proxies[trg], trg);
            }
        }

    };

    void show_phase(const std::vector<transfer>& transfers) const {
        for(const transfer& t : transfers) {
            std::cout <<   "src:" << t.src_gpu
                      << ", pos:" << t.src_pos
                      << ", trg:" << t.trg_gpu
                      << ", pos:" << t.trg_pos
                      << ", len:" << t.len
                      << std::endl;
        }
    }

    template<typename value_t>
    void execute_phase(value_t * srcs[num_gpus],
                       value_t * dsts[num_gpus],
                       const std::vector<transfer>& transfers) const {
        for(const transfer& t : transfers) {
            const gpu_id_t src = context->get_device_id(t.src_gpu);
            const gpu_id_t trg = context->get_device_id(t.trg_gpu);
            const auto stream  = context->get_streams(t.src_gpu)[t.trg_gpu];
            cudaSetDevice(src);
            const size_t size = t.len * sizeof(value_t);
            value_t * from = srcs[t.src_gpu] + t.src_pos;
            value_t * to   = dsts[t.trg_gpu] + t.trg_pos;

            cudaMemcpyPeerAsync(to, trg, from, src, size, stream);
        } CUERR
    }

    // only for convenience
    template <
        typename value_t,
        typename index_t>
    void clear(value_t * mem[num_gpus], index_t mem_lens[num_gpus]) {
        context->sync_all_streams();
        for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
            const gpu_id_t id = context->get_device_id(gpu);
            const auto stream = context->get_streams(gpu)[0];
            cudaSetDevice(id);
            const size_t size = mem_lens[gpu]
                              * sizeof(value_t);
            cudaMemsetAsync(mem[gpu], 0, size, stream);
        } CUERR
    }

public:
    template <
        typename value_t,
        typename index_t,
        typename table_t>
    bool execAsync (
        value_t * srcs[num_gpus],        // src[k] resides on device_ids[k]
        const index_t (&srcs_lens)[num_gpus],     // src_len[k] is length of src[k]
        value_t * dsts[num_gpus],        // dst[k] resides on device_ids[k]
        const index_t (&dsts_lens)[num_gpus],     // dst_len[0] is length of dst[k]
        const table_t (&table)[num_gpus][num_gpus]) const {  // [src_gpu, partition]

        // syncs with zero stream in order to enforce sequential
        // consistency with traditional synchronous memcpy calls
        if (!external_context)
            context->sync_hard();

        transfer_handler<table_t> transfers(table);

        transfers.one_to_all(0, {0,1,2,3,4,4,4,4});
        transfers.one_to_all(1, {0,1,2,3,5,5,5,5});
        transfers.one_to_all(2, {0,2,2,3,6,1,6,3});
        transfers.one_to_all(3, {3,1,2,3,0,7,2,7});
        transfers.one_to_all(4, {0,0,0,0,4,5,6,7});
        transfers.one_to_all(5, {1,1,1,1,4,5,6,7});
        transfers.one_to_all(6, {2,5,2,7,4,6,6,7});
        transfers.one_to_all(7, {4,3,6,3,7,5,6,7});

        // check if sufficient space for phase 1
        for (gpu_id_t trg = 0; trg < num_gpus; trg++) {
            if (transfers.phase_one_offsets[trg] > dsts_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "dsts_lens not compatible with partition_table.");
                else return false;
        }

        // check if sufficient space for phase 2
        for (gpu_id_t trg = 0; trg < num_gpus; trg++) {
            if (transfers.phase_two_offsets[trg] > srcs_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "srcs_lens not compatible with partition_table.");
                else return false;
        }


        /**********************************************************************
         * PHASE 1
         **********************************************************************/
        // show_phase(transfers.phase_one);
        execute_phase(srcs, dsts, transfers.phase_one);

        // only for convenience
        // clear(srcs, srcs_lens);

        // mandatory
        context->sync_all_streams();

        /**********************************************************************
         * PHASE 2
         **********************************************************************/
        // show_phase(transfers.phase_two);
        execute_phase(dsts, srcs, transfers.phase_two);

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

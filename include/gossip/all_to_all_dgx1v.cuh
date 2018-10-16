#pragma once

template<
    uint64_t num_gpus,
    uint64_t throw_exceptions=true>
class all2all_dgx1v_t {

    context_t<num_gpus> * context;
    bool external_context;

    static_assert(num_gpus==8, "currently only for exactly all GPUs.");
    
    struct transfer {
        const uint64_t src_gpu;
        const uint64_t src_pos;
        const uint64_t trg_gpu;
        const uint64_t trg_pos;
        const uint64_t len;

        transfer(const uint64_t src_gpu,
                 const uint64_t src_pos,
                 const uint64_t trg_gpu,
                 const uint64_t trg_pos,
                 const uint64_t len) :
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

        uint64_t phase_one_offsets[num_gpus] = {0};
        uint64_t phase_two_offsets[num_gpus] = {0};

        const table_t (&table)[num_gpus][num_gpus];
        uint64_t h_table[num_gpus][num_gpus+1] = {0}; // horizontal scan

        transfer_handler(const table_t (&table)[num_gpus][num_gpus]) : table(table) {
            for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
                for (uint64_t part = 0; part < num_gpus; ++part) {
                    h_table[gpu][part+1] = table[gpu][part]+h_table[gpu][part];
                }
            }
        }

        void push_back(const uint64_t src, const uint64_t proxy, const uint64_t trg) {
            const uint64_t transfer_size = table[src][trg];
            phase_one.emplace_back(src, h_table[src][trg], proxy, phase_one_offsets[proxy], transfer_size);
            phase_two.emplace_back(proxy, phase_one_offsets[proxy], trg, phase_two_offsets[trg], transfer_size);
            phase_one_offsets[proxy] += transfer_size;
            phase_two_offsets[trg] += transfer_size;
        }
    };

public:

    all2all_dgx1v_t (
        uint64_t * device_ids_=0) : external_context (false){

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
        
        //left quad
        for (uint64_t trg = 0; trg < num_gpus/2; trg++) {
            for (uint64_t src = 0; src < num_gpus/2; src++) {
                const uint64_t proxy = trg;
                transfers.push_back(src, proxy, trg);
            }
        }
        //right quad
        for (uint64_t trg = num_gpus/2; trg < num_gpus; trg++) {
            for (uint64_t src = num_gpus/2; src < num_gpus; src++) {
                const uint64_t proxy = trg;
                transfers.push_back(src, proxy, trg);
            }
        }
        //inner left to right
        for (uint64_t src = 0; src < 2; src++) {
            for(uint64_t trg = num_gpus/2; trg < num_gpus; trg++) {
                const uint64_t proxy = src+num_gpus/2;
                transfers.push_back(src, proxy, trg);
            }
        }
        //inner right to left
        for (uint64_t src = num_gpus/2; src < 2+num_gpus/2; src++) {
            for(uint64_t trg = 0; trg < num_gpus/2; trg++) {
                const uint64_t proxy = src-num_gpus/2;
                transfers.push_back(src, proxy, trg);
            }
        }
        //outer left to right
        uint64_t src, proxy, trg;
        {
            src = 2; proxy = 6; trg = 4;
            transfers.push_back(src, proxy, trg);

            src = 2; proxy = 1; trg = 5;
            transfers.push_back(src, proxy, trg);

            src = 2; proxy = 6; trg = 6;
            transfers.push_back(src, proxy, trg);

            src = 2; proxy = 3; trg = 7;
            transfers.push_back(src, proxy, trg);
        }
        {
            src = 3; proxy = 0; trg = 4;
            transfers.push_back(src, proxy, trg);

            src = 3; proxy = 7; trg = 5;
            transfers.push_back(src, proxy, trg);

            src = 3; proxy = 2; trg = 6;
            transfers.push_back(src, proxy, trg);

            src = 3; proxy = 7; trg = 7;
            transfers.push_back(src, proxy, trg);
        }
        //outer right to left
        {
            src = 6; proxy = 2; trg = 0;
            transfers.push_back(src, proxy, trg);

            src = 6; proxy = 5; trg = 1;
            transfers.push_back(src, proxy, trg);

            src = 6; proxy = 2; trg = 2;
            transfers.push_back(src, proxy, trg);

            src = 6; proxy = 7; trg = 3;
            transfers.push_back(src, proxy, trg);
        }
        {
            src = 7; proxy = 4; trg = 0;
            transfers.push_back(src, proxy, trg);

            src = 7; proxy = 3; trg = 1;
            transfers.push_back(src, proxy, trg);

            src = 7; proxy = 6; trg = 2;
            transfers.push_back(src, proxy, trg);

            src = 7; proxy = 3; trg = 3;
            transfers.push_back(src, proxy, trg);
        }
        
        // check if sufficient space for phase 1
        for (uint64_t trg = 0; trg < num_gpus; trg++) {
            if (transfers.phase_one_offsets[trg] > dsts_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "dsts_lens not compatible with partition_table.");
                else return false;
        }
 
        // check if sufficient space for phase 2
        for (uint64_t trg = 0; trg < num_gpus; trg++) {
            if (transfers.phase_two_offsets[trg] > srcs_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "srcs_lens not compatible with partition_table.");
                else return false;
        }


        /**********************************************************************
         * PHASE 1
         **********************************************************************/

        // for(const transfer& t : transfers.phase_one) {
        //     std::cout << "src:" << t.src_gpu
        //               << ", pos:" << t.src_pos
        //               << ", trg:" << t.trg_gpu
        //               << ", pos:" << t.trg_pos
        //               << ", len:" << t.len << std::endl;
        // }

        for(const transfer& t : transfers.phase_one) {
            const uint64_t src = context->get_device_id(t.src_gpu);
            const uint64_t trg = context->get_device_id(t.trg_gpu);
            const auto stream  = context->get_streams(t.src_gpu)[t.trg_gpu];
            cudaSetDevice(src);
            const uint64_t size = t.len * sizeof(value_t);
            value_t * from = srcs[t.src_gpu] + t.src_pos;
            value_t * to   = dsts[t.trg_gpu] + t.trg_pos;

            cudaMemcpyPeerAsync(to, trg, from, src, size, stream);
        } CUERR

        // only for convenience
        if (false) {
            context->sync_all_streams();
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; src_gpu++) {
                const uint64_t src = context->get_device_id(src_gpu);
                const auto stream  = context->get_streams(src_gpu)[0];
                cudaSetDevice(src);
                const uint64_t count = srcs_lens[src_gpu]
                                     * sizeof(value_t);
                cudaMemsetAsync(srcs[src_gpu], 0, count, stream);
            } CUERR
        }


        // mandatory
        context->sync_all_streams();


        /**********************************************************************
         * PHASE 2
         **********************************************************************/

        // for(const transfer& t : transfers.phase_two) {
        //     std::cout << "src:" << t.src_gpu
        //               << ", pos:" << t.src_pos
        //               << ", trg:" << t.trg_gpu
        //               << ", pos:" << t.trg_pos
        //               << ", len:" << t.len << std::endl;
        // }

         for(const transfer& t : transfers.phase_two) {
            const uint64_t src = context->get_device_id(t.src_gpu);
            const uint64_t trg = context->get_device_id(t.trg_gpu);
            const auto stream  = context->get_streams(t.src_gpu)[t.trg_gpu];
            cudaSetDevice(src);
            const uint64_t size = t.len * sizeof(value_t);
            value_t * from = dsts[t.src_gpu] + t.src_pos;
            value_t * to   = srcs[t.trg_gpu] + t.trg_pos;

            cudaMemcpyPeerAsync(to, trg, from, src, size, stream);
        } CUERR

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

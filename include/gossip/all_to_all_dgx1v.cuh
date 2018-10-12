#pragma once

template<
    uint64_t num_gpus,
    uint64_t throw_exceptions=true>
class all2all_dgx1v_t {

    context_t<num_gpus> * context;
    bool external_context;

    static_assert(num_gpus==8, "currently only for exactly all GPUs.");

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

        uint64_t UL[num_gpus/2+1][num_gpus/2];
        uint64_t UR[num_gpus/2+1][num_gpus/2];
        uint64_t LL[num_gpus/2+1][num_gpus/2];
        uint64_t LR[num_gpus/2+1][num_gpus/2];

        for (uint64_t trg = 0; trg < num_gpus/2; trg++) {
            UL[0][trg] = 0;
            UR[0][trg] = 0;
            LL[0][trg] = 0;
            LR[0][trg] = 0;
        }

        for (uint64_t src = 0; src < num_gpus/2; src++) {
            for (uint64_t trg = 0; trg < num_gpus/2; trg++) {
                UL[src+1][trg] = UL[src][trg]
                               + table[src][trg];
                UR[src+1][trg] = UR[src][trg]
                               + table[src][trg+num_gpus/2];
                LL[src+1][trg] = LL[src][trg]
                               + table[src+num_gpus/2][trg];
                LR[src+1][trg] = LR[src][trg]
                               + table[src+num_gpus/2][trg+num_gpus/2];
            }
        }

        // std::cout << "UL" << std::endl;
        // for (uint64_t src = 0; src < num_gpus/2+1; src++)
        //     for (uint64_t trg = 0; trg < num_gpus/2; trg++)
        //         std::cout << UL[src][trg] << (trg == num_gpus/2-1 ? "\n" : " ");
        // std::cout << std::endl;

        // std::cout << "UR" << std::endl;
        // for (uint64_t src = 0; src < num_gpus/2+1; src++)
        //     for (uint64_t trg = 0; trg < num_gpus/2; trg++)
        //         std::cout << UR[src][trg] << (trg == num_gpus/2-1 ? "\n" : " ");
        // std::cout << std::endl;

        // std::cout << "LL" << std::endl;
        // for (uint64_t src = 0; src < num_gpus/2+1; src++)
        //     for (uint64_t trg = 0; trg < num_gpus/2; trg++)
        //         std::cout << LL[src][trg] << (trg == num_gpus/2-1 ? "\n" : " ");
        // std::cout << std::endl;

        // std::cout << "LR" << std::endl;
        // for (uint64_t src = 0; src < num_gpus/2+1; src++)
        //     for (uint64_t trg = 0; trg < num_gpus/2; trg++)
        //         std::cout << LR[src][trg] << (trg == num_gpus/2-1 ? "\n" : " ");
        // std::cout << std::endl;


        // check if sufficient space for phase 1
        for (uint64_t trg = 0; trg < num_gpus/2; trg++) {
            if (UL[num_gpus/2][trg] + UR[num_gpus/2][trg] > dsts_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "dsts_lens not compatible with partition_table.");
                else return false;
            if (LL[num_gpus/2][trg] + LR[num_gpus/2][trg] > dsts_lens[trg+num_gpus/2])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "dsts_lens not compatible with partition_table.");
                else return false;
        }

        // check if sufficient space for phase 2
        for (uint64_t trg = 0; trg < num_gpus/2; trg++) {
            if (UL[num_gpus/2][trg] + LL[num_gpus/2][trg] > srcs_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "srcs_lens not compatible with partition_table.");
                else return false;
            if (UR[num_gpus/2][trg] + LR[num_gpus/2][trg] > srcs_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "srcs_lens not compatible with partition_table.");
                else return false;
        }


        /**********************************************************************
         * PHASE 1
         **********************************************************************/

        // upper subgroup
        for (uint64_t src_gpu = 0; src_gpu < num_gpus/2; src_gpu++) {
            const uint64_t src = context->get_device_id(src_gpu);
            cudaSetDevice(src);
            // left subgroup
            for (uint64_t trg_gpu = 0; trg_gpu < num_gpus/2; trg_gpu++) {
                const uint64_t trg = context->get_device_id(trg_gpu);            

                uint64_t count = table[src_gpu][trg_gpu]
                               * sizeof(value_t);
                value_t * to   = dsts [trg_gpu]
                               + UL[src_gpu][trg_gpu];
                value_t * from = srcs [src_gpu]
                               + h_table[src_gpu][trg_gpu];

                cudaMemcpyPeerAsync(to, trg, from, src, count,
                                    context->get_streams(src_gpu)[trg_gpu]);
            }
            // right subgroup
            for (uint64_t trg_gpu = 0; trg_gpu < num_gpus/2; trg_gpu++) {
                const uint64_t trg = context->get_device_id(trg_gpu);

                uint64_t count = table[src_gpu][trg_gpu+num_gpus/2]
                               * sizeof(value_t);
                value_t * to   = dsts [trg_gpu]
                               + UL[num_gpus/2][trg_gpu]
                               + UR[src_gpu][trg_gpu];
                value_t * from = srcs [src_gpu]
                               + h_table[src_gpu][trg_gpu+num_gpus/2];

                cudaMemcpyPeerAsync(to, trg, from, src, count,
                                    context->get_streams(src_gpu)[trg_gpu+num_gpus/2]);
            }
        } CUERR

        // lower subgroup
        for (uint64_t src_gpu = 0; src_gpu < num_gpus/2; src_gpu++) {
            const uint64_t src = context->get_device_id(src_gpu+num_gpus/2);
            cudaSetDevice(src);
            // left subgroup
            for (uint64_t trg_gpu = 0; trg_gpu < num_gpus/2; trg_gpu++) {
                const uint64_t trg = context->get_device_id(trg_gpu+num_gpus/2);            

                uint64_t count = table[src_gpu+num_gpus/2][trg_gpu]
                               * sizeof(value_t);
                value_t * to   = dsts [trg_gpu+num_gpus/2]
                               + LL[src_gpu][trg_gpu];
                value_t * from = srcs [src_gpu+num_gpus/2]
                               + h_table[src_gpu+num_gpus/2][trg_gpu];

                cudaMemcpyPeerAsync(to, trg, from, src, count,
                                    context->get_streams(src_gpu+num_gpus/2)[trg_gpu]);

            }
            // right subgroup
            for (uint64_t trg_gpu = 0; trg_gpu < num_gpus/2; trg_gpu++) {
                const uint64_t trg = context->get_device_id(trg_gpu+num_gpus/2);

                uint64_t count = table[src_gpu+num_gpus/2][trg_gpu+num_gpus/2]
                               * sizeof(value_t);
                value_t * to   = dsts [trg_gpu+num_gpus/2]
                               + LL[num_gpus/2][trg_gpu]
                               + LR[src_gpu][trg_gpu];
                value_t * from = srcs [src_gpu+num_gpus/2]
                               + h_table[src_gpu+num_gpus/2][trg_gpu+num_gpus/2];

                cudaMemcpyPeerAsync(to, trg, from, src, count,
                                    context->get_streams(src_gpu+num_gpus/2)[trg_gpu+num_gpus/2]);
            }
        } CUERR


        // only for convenience
	if (false) {
	    context->sync_all_streams();
            for (uint64_t src_gpu = 0; src_gpu < num_gpus; src_gpu++) {
                const uint64_t src = context->get_device_id(src_gpu);
                cudaSetDevice(src);
                const uint64_t count = srcs_lens[src_gpu]
                                     * sizeof(value_t);
                cudaMemsetAsync(srcs[src_gpu], 0, count,
                                context->get_streams(src_gpu)[0]);
            } CUERR
        }


        // mandatory
        context->sync_all_streams();


        /**********************************************************************
         * PHASE 2
         **********************************************************************/

        // upper left subgroup
        for (uint64_t gpu = 0; gpu < num_gpus/2; ++gpu) {
            const uint64_t src_gpu = gpu;
            const uint64_t trg_gpu = gpu;
            const uint64_t src = context->get_device_id(src_gpu);
            const uint64_t trg = context->get_device_id(trg_gpu);
            cudaSetDevice(src);
            const uint64_t len = UL[num_gpus/2][gpu];
            value_t * from = dsts[src_gpu];
            value_t * to   = srcs[trg_gpu];

            cudaMemcpyAsync(to, from, len*sizeof(value_t), cudaMemcpyDeviceToDevice,
                            context->get_streams(src_gpu)[trg_gpu]);
        } CUERR

        // lower right subgroup
        for (uint64_t gpu = 0; gpu < num_gpus/2; ++gpu) {
            const uint64_t src_gpu = gpu+num_gpus/2;
            const uint64_t trg_gpu = gpu+num_gpus/2;
            const uint64_t src = context->get_device_id(src_gpu);
            const uint64_t trg = context->get_device_id(trg_gpu);
            cudaSetDevice(src);
            const uint64_t len = LR[num_gpus/2][gpu];
            value_t * from = dsts[src_gpu]
                           + LL[num_gpus/2][gpu];
            value_t * to   = srcs[trg_gpu]
                           + UR[num_gpus/2][gpu];

            cudaMemcpyAsync(to, from, len*sizeof(value_t), cudaMemcpyDeviceToDevice,
                            context->get_streams(src_gpu)[trg_gpu]);
        } CUERR

        // upper right subgroup
        for (uint64_t gpu = 0; gpu < num_gpus/2; ++gpu) {
            const uint64_t src_gpu = gpu;
            const uint64_t trg_gpu = gpu+num_gpus/2;
            const uint64_t src = context->get_device_id(src_gpu);
            const uint64_t trg = context->get_device_id(trg_gpu);
            cudaSetDevice(src);
            const uint64_t len = UR[num_gpus/2][gpu];
            value_t * from = dsts[src_gpu]
                           + UL[num_gpus/2][gpu];
            value_t * to   = srcs[trg_gpu];

            cudaMemcpyPeerAsync(to, trg, from, src,
                                len*sizeof(value_t),
                                context->get_streams(src_gpu)[trg_gpu]);
        } CUERR

        // lower left subgroup
        for (uint64_t gpu = 0; gpu < num_gpus/2; ++gpu) {
            const uint64_t src_gpu = gpu+num_gpus/2;
            const uint64_t trg_gpu = gpu;
            const uint64_t src = context->get_device_id(src_gpu);
            const uint64_t trg = context->get_device_id(trg_gpu);
            cudaSetDevice(src);
            const uint64_t len = LL[num_gpus/2][gpu];
            value_t * from = dsts[src_gpu];
            value_t * to   = srcs[trg_gpu]
                           + UL[num_gpus/2][gpu];

            cudaMemcpyPeerAsync(to, trg, from, src,
                                len*sizeof(value_t),
                                context->get_streams(src_gpu)[trg_gpu]);
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

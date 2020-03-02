#pragma once

#include <vector>

#include "config.h"
#include "error_checking.hpp"
#include "context.cuh"
#include "broadcast_plan.hpp"

namespace gossip {

class broadcast_t {

private:
    const context_t * context;

    transfer_plan_t transfer_plan;
    bool plan_valid;

public:
     broadcast_t (
        const context_t& context_,
        const gpu_id_t main_gpu_)
        : context(&context_),
          transfer_plan( broadcast::default_plan(context->get_num_devices(), main_gpu_) ),
          plan_valid( transfer_plan.valid() )
    {
        check(context->is_valid(),
              "You have to pass a valid context!");
    }

    broadcast_t (
        const context_t& context_,
        const transfer_plan_t& transfer_plan_)
        : context(&context_),
          transfer_plan(transfer_plan_),
          plan_valid(false)
    {
        check(context->is_valid(),
              "You have to pass a valid context!");

        if(!transfer_plan.valid())
            broadcast::verify_plan(transfer_plan);

        check(get_num_devices() == transfer_plan.num_gpus(),
              "Plan does fit number of gpus of context!");

        plan_valid = (get_num_devices() == transfer_plan.num_gpus()) &&
                     transfer_plan.valid();
    }

    void show_plan() const {
        if(!plan_valid)
            std::cout << "WARNING: plan does fit number of gpus\n";

        transfer_plan.show_plan();
    }

private:
    struct transfer {
        size_t src_pos;
        size_t trg_pos;
        size_t len;
        cudaEvent_t* event_before;
        cudaEvent_t* event_after;

        transfer(
            cudaEvent_t* event_before,
            cudaEvent_t* event_after
        ) :
            event_before(event_before),
            event_after(event_after)
        {}

        ~transfer() {
            // if(event_before) cudaEventDestroy(*event_before);
            if(event_after) cudaEventDestroy(*event_after);
        }

        void set_pos(
            const size_t src_pos_,
            const size_t trg_pos_,
            const size_t len_
        ) {
            src_pos = src_pos_;
            trg_pos = trg_pos_;
            len = len_;
        }

        void show(const gpu_id_t src_gpu, const gpu_id_t trg_gpu) const {
            std::cout <<   "src:" << uint32_t(src_gpu)
                      << ", pos:" << src_pos
                      << ", trg:" << uint32_t(trg_gpu)
                      << ", pos:" << trg_pos
                      << ", len:" << len
                      << std::endl;
        }
    };

    struct transfer_handler {
        const context_t * context;
        const transfer_plan_t& transfer_plan;
        std::vector<std::vector<transfer *> > transfers;

        transfer_handler(
            const context_t * context,
            const transfer_plan_t& transfer_plan
        ) :
            context(context),
            transfer_plan(transfer_plan),
            transfers(num_steps(), std::vector<transfer *>(num_gpus()*num_gpus()))
        {
            load_plan();
        }

        ~transfer_handler() {
            for(auto& transfer_step : transfers) {
                for(auto& t : transfer_step) {
                    if(t) delete(t);
                }
            }
        }

        void load_plan() {
            // on main gpu: direct transfer (copy) in first step
            {
                const size_t step = 0;

                const gpu_id_t src = transfer_plan.main_gpu();
                const gpu_id_t trg = transfer_plan.main_gpu();
                const size_t index = src * num_gpus() + trg;

                cudaEvent_t* event_before = nullptr;
                cudaEvent_t* event_after = nullptr;

                transfers[step][index] = new transfer(event_before, event_after);
            }

            // other gpus: transfer according to sequence
            for(const auto& sequence : transfer_plan.transfer_sequences()) {

                const auto final_trg = sequence.seq.back();

                if (sequence.seq.front() != final_trg) {
                    cudaEvent_t* event_before = nullptr;
                    cudaEvent_t* event_after = nullptr;

                    for(size_t step = 0; step < num_steps(); ++step) {
                        const gpu_id_t src = sequence.seq[step];
                        const gpu_id_t trg = sequence.seq[step+1];

                        // create transfer only if device changes
                        if(src != trg) {
                            const size_t index = src * num_gpus() + trg;

                            if(!transfers[step][index]) {
                                // create transfer if it does not exist
                                if(trg != final_trg) {
                                    // create event after transfer for synchronization
                                    event_after = new cudaEvent_t();
                                    const gpu_id_t id = context->get_device_id(src);
                                    cudaSetDevice(id);
                                    cudaEventCreate(event_after);
                                }
                                else {
                                    // final transfer does not need follow up event
                                    event_after = nullptr;
                                }

                                transfers[step][index] = new transfer(event_before, event_after);

                                event_before = event_after;
                            }
                            else {
                                // transfer already exists, use its event for synchronization
                                event_before = transfers[step][index]->event_after;
                            }
                        }
                    }
                }
            }
        }

        gpu_id_t num_gpus() const noexcept {
            return transfer_plan.num_gpus();
        }

        gpu_id_t num_steps() const noexcept {
            return transfer_plan.num_steps();
        }

        gpu_id_t num_chunks() const noexcept {
            return transfer_plan.num_chunks();
        }

        template<typename table_t>
        void calculate_offsets(
            const std::vector<size_t>& src_displacements,
            const std::vector<table_t>& sizes,
            const bool verbose = false
        ) {
            std::vector<size_t> src_offsets(src_displacements);

            // on main gpu: direct transfer (copy) in first step
            {
                const gpu_id_t src = transfer_plan.main_gpu();
                const gpu_id_t trg = transfer_plan.main_gpu();
                const size_t index = src * num_gpus() + trg;

                const size_t step = 0;
                const size_t transfer_size = src_displacements.back();
                transfers[step][index]->set_pos(0, 0, transfer_size);

                if (verbose) transfers[step][index]->show(src, trg);
            }

            // other gpus: transfer according to sequence
            for(const auto& sequence : transfer_plan.transfer_sequences()) {

                if (verbose)
                    std::cout << "transfer from " << int(sequence.seq.front())
                              << " to " << int(sequence.seq.back())
                              << std::endl;

                if (sequence.seq.front() != sequence.seq.back()) {
                    const size_t chunk_id = sequence.size;
                    const size_t src_offset = src_offsets[chunk_id];
                    const size_t trg_offset = src_offset;
                    const size_t transfer_size = sizes[chunk_id];

                    for(size_t step = 0; step < num_steps(); ++step) {
                        const gpu_id_t src = sequence.seq[step];
                        const gpu_id_t trg = sequence.seq[step+1];

                        // transfer only if device changes
                        if(src != trg) {
                            const size_t index = src * num_gpus() + trg;

                            if(transfers[step][index]->len == 0) {
                                transfers[step][index]->set_pos(src_offset, trg_offset, transfer_size);
                            }
                            if (verbose) transfers[step][index]->show(src, trg);
                        }
                    }
                }
            }
        }

        void show_step(const size_t step) const {
            for(gpu_id_t src = 0; src < num_gpus(); ++src) {
                for(gpu_id_t trg = 0; trg < num_gpus(); ++trg) {
                    const size_t index = src * num_gpus() + trg;
                    if(transfers[step][index])
                        transfers[step][index]->show(src, trg);
                }
            }
        }

        void show_steps() const {
            for(size_t step = 0; step < num_steps(); ++step)
                show_step(step);
        }

        template<typename value_t>
        void execute_step(
            const size_t step,
            const value_t * sendbuf,
            const std::vector<value_t *>& recvbufs
        ) {
            for(gpu_id_t t_src_gpu = 0; t_src_gpu < num_gpus(); ++t_src_gpu) {
                for(gpu_id_t t_trg_gpu = 0; t_trg_gpu < num_gpus(); ++t_trg_gpu) {
                    const size_t index = t_src_gpu * num_gpus() + t_trg_gpu;
                    transfer * t = transfers[step][index];
                    if(t && t->len > 0) {
                        const gpu_id_t src = context->get_device_id(t_src_gpu);
                        const gpu_id_t trg = context->get_device_id(t_trg_gpu);
                        const auto stream  = context->get_streams(t_src_gpu)[t_trg_gpu];
                        cudaSetDevice(src);
                        const size_t size = t->len * sizeof(value_t);
                        const value_t * from = (t->event_before == nullptr) ?
                                            sendbuf + t->src_pos :
                                            recvbufs[t_src_gpu] + t->src_pos;
                        value_t * to   = recvbufs[t_trg_gpu] + t->trg_pos;

                        if(t->event_before != nullptr) cudaStreamWaitEvent(stream, *(t->event_before), 0);
                        cudaMemcpyPeerAsync(to, trg, from, src, size, stream);
                        if(t->event_after != nullptr) cudaEventRecord(*(t->event_after), stream);

                        t->len = 0;
                    } CUERR
                }
            }
        }

        template<typename value_t>
        void execute_steps(
            const value_t * sendbuf,
            const std::vector<value_t *>& recvbufs
        ) {
            for(size_t step = 0; step < num_steps(); ++step)
                execute_step(step, sendbuf, recvbufs);
        }
    };

public:
    template <
        typename value_t,
        typename index_t>
    bool execAsync (
        const value_t * sendbuf,                     // sendbuf resides on device_ids[main_gpu]
        const index_t sendbuf_len,                   // sendbuf_len is length of sendbuf
        const size_t sendsize,     // send sendsizes[k] bytes to device_ids[k]
        const std::vector<value_t *>& recvbufs,      // recvbufs[k] resides on device_ids[k]
        const std::vector<index_t  >& recvbufs_lens  // recvbufs_len[k] is length of recvbufs[k]
    ) const {

        if (!plan_valid) return false;

        if (!check(recvbufs.size() == get_num_devices(),
                    "recvbufs size does not match number of gpus."))
            return false;
        if (!check(recvbufs_lens.size() == get_num_devices(),
                    "recvbufs_lens size does not match number of gpus."))
            return false;

        const auto num_steps = transfer_plan.num_steps();
        const auto num_chunks = transfer_plan.num_chunks();

        size_t size_per_chunk = SDIV(sendsize, num_chunks);
        std::vector<size_t> sendsizes(num_chunks, size_per_chunk);
        sendsizes.back() = sendsize - (num_chunks-1) * size_per_chunk;

        std::vector<size_t> displacements(num_chunks+1);

        for (gpu_id_t part = 0; part < num_chunks; ++part) {
            // exclusive scan to get displacements
            displacements[part+1] = displacements[part] + sendsizes[part];
        }
        if(displacements.back() != sendsize) {
            std::cerr << "sendsizes sum is wrong" << std::endl;
        }

        transfer_handler transfers(context, transfer_plan);

        bool verbose = false;
        // prepare transfers according to transfer_plan
        transfers.calculate_offsets(displacements, sendsizes, verbose);

        transfers.show_steps();

        // if (!check_sendbuf_size(displacements.back(), sendbuf_len))
        //     return false;
        // if (!check_transfers_size(transfers.trg_offsets, transfers.aux_offsets, recvbufs_lens))
        //     return false;

        transfers.execute_steps(sendbuf, recvbufs);

        return true;
    }

    gpu_id_t get_num_devices () const noexcept {
        return context->get_num_devices();
    }

    void sync () const noexcept {
        context->sync_all_streams();
    }

    void sync_hard () const noexcept {
        context->sync_hard();
    }

    const context_t& get_context() const noexcept {
        return *context;
    }
};

} // namespace
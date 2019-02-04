#pragma once

#include <vector>
#include <stdexcept>

#include "config.h"
#include "context.cuh"
#include "scatter_plan.hpp"

namespace gossip {

template<
    bool throw_exceptions=true>
class scatter_t {

private:
    const context_t<> * context;
    bool external_context;

    transfer_plan_t transfer_plan;
    bool plan_valid;

public:
    scatter_t (
        const gpu_id_t num_gpus_,
        const gpu_id_t main_gpu_)
        : external_context (false)
    {
        context = new context_t<>(num_gpus_);

        transfer_plan = scatter::default_plan(num_gpus_, main_gpu_);

        plan_valid = transfer_plan.valid();
    }

    scatter_t (
        const gpu_id_t num_gpus_,
        const transfer_plan_t& transfer_plan_)
        : external_context (false),
          transfer_plan(transfer_plan_)
    {
        context = new context_t<>(num_gpus_);

        if(!transfer_plan.valid())
            scatter::verify_plan(transfer_plan);

        plan_valid = (get_num_devices() == transfer_plan.num_gpus()) &&
                     transfer_plan.valid();
    }

    scatter_t (
        const std::vector<gpu_id_t>& device_ids_,
        const gpu_id_t main_gpu_)
        : external_context (false)
    {
        context = new context_t<>(device_ids_);

        transfer_plan = scatter::default_plan(device_ids_.size(), main_gpu_);

        plan_valid = transfer_plan.valid();
    }

    scatter_t (
        const std::vector<gpu_id_t>& device_ids_,
        const transfer_plan_t& transfer_plan_)
        : external_context (false),
          transfer_plan(transfer_plan_)
    {
        context = new context_t<>(device_ids_);

        if(!transfer_plan.valid())
            scatter::verify_plan(transfer_plan);

        plan_valid = (get_num_devices() == transfer_plan.num_gpus()) &&
                     transfer_plan.valid();
    }

     scatter_t (
        const context_t<> * context_,
        const gpu_id_t main_gpu_)
        : context(context_),
          external_context (true)
    {
        if (throw_exceptions)
            if (!context->is_valid())
                throw std::invalid_argument(
                    "You have to pass a valid context!"
                );

        transfer_plan = scatter::default_plan(get_num_devices(), main_gpu_);

        plan_valid = transfer_plan.valid();
    }

    scatter_t (
        const context_t<> * context_,
        const transfer_plan_t& transfer_plan_)
        : context(context_),
          external_context (true),
          transfer_plan(transfer_plan_)
    {
        if (throw_exceptions)
            if (!context->is_valid())
                throw std::invalid_argument(
                    "You have to pass a valid context!"
                );

        if(!transfer_plan.valid())
            scatter::verify_plan(transfer_plan);

        plan_valid = (get_num_devices() == transfer_plan.num_gpus()) &&
                     transfer_plan.valid();
    }

    ~scatter_t () {
        if (!external_context)
            delete context;
    }

public:
    void show_plan() const {
        if(!plan_valid)
            std::cout << "WARNING: plan does fit number of gpus\n";

        transfer_plan.show_plan();
    }

private:
    struct transfer {
        const gpu_id_t src_gpu;
        const size_t src_pos;
        const gpu_id_t trg_gpu;
        const size_t trg_pos;
        const size_t len;
        const cudaEvent_t* event_before;
        const cudaEvent_t* event_after;

        transfer(const gpu_id_t src_gpu,
                 const size_t src_pos,
                 const gpu_id_t trg_gpu,
                 const size_t trg_pos,
                 const size_t len,
                 const cudaEvent_t* event_before,
                 const cudaEvent_t* event_after) :
            src_gpu(src_gpu),
            src_pos(src_pos),
            trg_gpu(trg_gpu),
            trg_pos(trg_pos),
            len(len),
            event_before(event_before),
            event_after(event_after)
        {}

        void show() const {
            std::cout <<   "src:" << int(src_gpu)
                      << ", pos:" << src_pos
                      << ", trg:" << int(trg_gpu)
                      << ", pos:" << trg_pos
                      << ", len:" << len
                      << ", event before:" << (event_before ? event_before : 0)
                      << ", event after:" << (event_after ? event_after : 0)
                      << std::endl;
        }
    };

    template<typename table_t>
    struct transfer_handler {
        const context_t<> * context;
        size_t num_phases;
        std::vector<std::vector<transfer> > phases;
        std::vector<size_t> src_offsets;
        std::vector<size_t> trg_offsets;
        std::vector<size_t> aux_offsets;

        const std::vector<size_t>& src_displacements;
        const std::vector<table_t>& total_sizes;

        size_t num_chunks;

        std::vector<cudaEvent_t*> events;

        transfer_handler(
            const context_t<> * context_,
            const size_t num_phases_,
            const std::vector<size_t>& src_displacements,
            const std::vector<table_t>& total_sizes,
            const size_t num_chunks_ = 1
        ) :
            context(context_),
            num_phases(num_phases_),
            phases(num_phases),
            // src offsets begin at src displacements
            src_offsets(src_displacements),
            trg_offsets(context->get_num_devices()),
            // aux offsets begin at the end of own part
            aux_offsets(total_sizes),
            src_displacements(src_displacements),
            total_sizes(total_sizes),
            num_chunks(num_chunks_)
        {}

        ~transfer_handler() {
            for(auto& e : events)
                cudaEventDestroy(*e);
        }

        bool push_back(
            const std::vector<gpu_id_t>& sequence,
            const size_t chunks = 1,
            const bool verbose = false)
        {
            if (sequence.size() != num_phases+1)
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "sequence size does not match number of phases.");
                else return false;

            size_t* src_offset = &src_offsets[sequence.back()];
            const size_t size_per_chunk = SDIV(total_sizes[sequence.back()], num_chunks);
            size_t transfer_size = size_per_chunk * chunks;
            // check bounds
            const size_t limit = src_displacements[sequence.back()] + total_sizes[sequence.back()];
            if (*src_offset + transfer_size > limit)
                transfer_size = limit - *src_offset;

            size_t* trg_offset = nullptr;

            cudaEvent_t* event_before = nullptr;
            cudaEvent_t* event_after = nullptr;

            if (verbose)
                std::cout << "transfer from " << int(sequence.front())
                          << " to " << int(sequence.back())
                          << std::endl;

            if (sequence.front() == sequence.back()) { // src == trg
                // direct transfer (copy) in first phase
                const size_t phase = 0;
                trg_offset = &trg_offsets[sequence.back()];

                phases[phase].emplace_back(sequence.front(), *src_offset,
                                       sequence.back(), *trg_offset,
                                       transfer_size,
                                       event_before, event_after);
                if (verbose) phases[phase].back().show();

                // advance offsets
                *src_offset += transfer_size;
                *trg_offset += transfer_size;
            }
            else { // src != trg
                const gpu_id_t final_trg = sequence.back();

                for (size_t phase = 0; phase < num_phases; ++phase) {
                    // schedule transfer only if device changes
                    if (sequence[phase] != sequence[phase+1]) {
                        if (sequence[phase+1] != final_trg) {
                            // transfer to auxiliary memory
                            trg_offset = &aux_offsets[sequence[phase+1]];
                            // create event after transfer for synchronization
                            event_after = new cudaEvent_t();
                            const gpu_id_t id = context->get_device_id(sequence[phase]);
                            cudaSetDevice(id);
                            cudaEventCreate(event_after);
                            events.push_back(event_after);
                        }
                        else {
                            // transfer to final memory position
                            trg_offset = &trg_offsets[sequence[phase+1]];
                            // final transfer does not need follow up event
                            event_after = nullptr;
                        }

                        phases[phase].emplace_back(sequence[phase], *src_offset,
                                                   sequence[phase+1], *trg_offset,
                                                   transfer_size,
                                                   event_before, event_after);
                        if (verbose) phases[phase].back().show();

                        // advance offset
                        *src_offset += transfer_size;
                        // old target is new source
                        src_offset = trg_offset;
                        event_before = event_after;

                        if (sequence[phase+1] == final_trg)
                            break;
                    }
                }
                // advance last offset
                if (trg_offset)
                    *trg_offset += transfer_size;
            }

            return true;
        }
    };

    void show_phase(const std::vector<transfer>& transfers) const {
        for(const transfer& t : transfers) {
            t.show();
        }
    }

    template<typename index_t>
    bool check_sendbuf_size(const size_t buf_accessed,
                            const index_t buf_len
    ) const {
        if (buf_accessed > buf_len)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "sendbuf access out of bounds.");
            else return false;
        return true;
    }

    template<typename index_t>
    bool check_transfers_size(
        const std::vector<size_t >& own_sizes,
        const std::vector<size_t >& buffer_sizes,
        const std::vector<index_t>& array_lens
    ) const {
        for (gpu_id_t trg = 0; trg < get_num_devices(); trg++) {
            if (own_sizes[trg] > array_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "recvbuf not large enough for results.");
                else return false;

            if (buffer_sizes[trg] > array_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "recvbuf not large enough for results + buffer overhead.");
                else return false;
        }
        return true;
    }

    template<typename value_t>
    bool execute_phase(const std::vector<transfer>& transfers,
                       const value_t * sendbuf,
                       const std::vector<value_t *>& recvbufs
    ) const {
        if (recvbufs.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "recvbufs size does not match number of gpus.");
            else return false;

        for(const transfer& t : transfers) {
            const gpu_id_t src = context->get_device_id(t.src_gpu);
            const gpu_id_t trg = context->get_device_id(t.trg_gpu);
            const auto stream  = context->get_streams(t.src_gpu)[t.trg_gpu];
            cudaSetDevice(src);
            const size_t size = t.len * sizeof(value_t);
            const value_t * from = (t.src_gpu == transfer_plan.main_gpu()) ?
                                   sendbuf + t.src_pos :
                                   recvbufs[t.src_gpu] + t.src_pos;
            value_t * to   = recvbufs[t.trg_gpu] + t.trg_pos;

            if(t.event_before != nullptr) cudaStreamWaitEvent(stream, *(t.event_before), 0);
            cudaMemcpyPeerAsync(to, trg, from, src, size, stream);
            if(t.event_after != nullptr) cudaEventRecord(*(t.event_after), stream);
        } CUERR

        return true;
    }

public:
    template <
        typename value_t,
        typename index_t,
        typename table_t>
    bool execAsync (
        const value_t * sendbuf,                     // sendbuf resides on device_ids[main_gpu]
        const index_t sendbuf_len,                   // sendbuf_len is length of sendbuf
        const std::vector<table_t  >& sendsizes,     // send sendsizes[k] bytes to device_ids[k]
        const std::vector<value_t *>& recvbufs,      // recvbufs[k] resides on device_ids[k]
        const std::vector<index_t  >& recvbufs_lens  // recvbufs_len[k] is length of recvbufs[k]
    ) const {

        if (!plan_valid) return false;

        if (recvbufs.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "recvbufs size does not match number of gpus.");
            else return false;
        if (recvbufs_lens.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "recvbufs_lens size does not match number of gpus.");
            else return false;
        if (sendsizes.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "sendsizes size does not match number of gpus.");
            else return false;

        const auto num_phases = transfer_plan.num_steps();
        const auto num_chunks = transfer_plan.num_chunks();

        std::vector<size_t> displacements(get_num_devices()+1);
        for (gpu_id_t part = 0; part < get_num_devices(); ++part) {
            // exclusive scan to get displacements
            displacements[part+1] = sendsizes[part] + displacements[part];
        }

        transfer_handler<table_t> transfers(context, num_phases,
                                            displacements, sendsizes,
                                            num_chunks);

        bool verbose = false;
        // prepare transfers according to transfer_plan
        for (const auto& sequence : transfer_plan.transfer_sequences()) {
            transfers.push_back(sequence.seq, sequence.size, verbose);
        }

        // for (size_t p = 0; p < num_phases; ++p)
        //     show_phase(transfers.phases[p]);

        if (!check_sendbuf_size(displacements.back(), sendbuf_len))
            return false;
        if (!check_transfers_size(transfers.trg_offsets, transfers.aux_offsets, recvbufs_lens))
            return false;

        // syncs with zero stream in order to enforce sequential
        // consistency with traditional synchronous memcpy calls
        if (!external_context)
            context->sync_hard();

        for (size_t p = 0; p < num_phases; ++p) {
            execute_phase(transfers.phases[p], sendbuf, recvbufs);
        }

        return true;
    }

    gpu_id_t get_num_devices () const noexcept {
        return context->get_num_devices();
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

} // namespace

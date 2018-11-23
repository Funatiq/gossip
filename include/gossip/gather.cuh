#pragma once

#include <vector>

#include "config.h"
#include "gather_plan.hpp"

namespace gossip {

template<
    bool throw_exceptions=true>
class gather_t {

protected:
    context_t<> * context;
    gpu_id_t num_gpus;
private:
    bool external_context;

    const gather_plan_t<> transfer_plan;

    bool plan_valid;

public:
    gather_t (
        const gpu_id_t num_gpus_,
        const gpu_id_t main_gpu_)
        : external_context (false),
          transfer_plan(main_gpu_, num_gpus_)
    {
        context = new context_t<>(num_gpus_);
        num_gpus = context->get_num_devices();

        plan_valid = transfer_plan.is_valid();
    }

    gather_t (
        const gpu_id_t num_gpus_,
        const gather_plan_t<>& transfer_plan_)
        : external_context (false),
          transfer_plan(transfer_plan_)
    {
        context = new context_t<>(num_gpus_);
        num_gpus = context->get_num_devices();

        plan_valid = (num_gpus == transfer_plan.get_num_gpus()) &&
                     transfer_plan.is_valid();
    }

    gather_t (
        const std::vector<gpu_id_t>& device_ids_,
        const gpu_id_t main_gpu_)
        : external_context (false),
          transfer_plan(main_gpu_, device_ids_.size())
    {
        context = new context_t<>(device_ids_);
        num_gpus = context->get_num_devices();

        plan_valid = transfer_plan.is_valid();
    }

    gather_t (
        const std::vector<gpu_id_t>& device_ids_,
        const gather_plan_t<>& transfer_plan_)
        : external_context (false),
          transfer_plan(transfer_plan_)
    {
        context = new context_t<>(device_ids_);
        num_gpus = context->get_num_devices();

        plan_valid = (num_gpus == transfer_plan.get_num_gpus()) &&
                     transfer_plan.is_valid();
    }

     gather_t (
        context_t<> * context_,
        const gpu_id_t main_gpu_)
        : context(context_),
          num_gpus(context->get_num_devices()),
          external_context (true),
          transfer_plan(main_gpu_, num_gpus)
    {
        if (throw_exceptions)
            if (!context->is_valid())
                throw std::invalid_argument(
                    "You have to pass a valid context!"
                );

        plan_valid = transfer_plan.is_valid();
    }

    gather_t (
        context_t<> * context_,
        const gather_plan_t<>& transfer_plan_)
        : context(context_),
          num_gpus(context->get_num_devices()),
          external_context (true),
          transfer_plan(transfer_plan_)
    {
        if (throw_exceptions)
            if (!context->is_valid())
                throw std::invalid_argument(
                    "You have to pass a valid context!"
                );

        plan_valid = (num_gpus == transfer_plan.get_num_gpus()) &&
                     transfer_plan.is_valid();
    }

    ~gather_t () {
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

        void show() {
            std::cout <<   "src:" << int(src_gpu)
                      << ", pos:" << src_pos
                      << ", trg:" << int(trg_gpu)
                      << ", pos:" << trg_pos
                      << ", len:" << len
                      << std::endl;
        }
    };

    template<typename table_t>
    struct transfer_handler {
        gpu_id_t num_gpus;
        size_t num_phases;
        std::vector<std::vector<transfer> > phases;
        std::vector<size_t> trg_offsets;
        std::vector<size_t> own_offsets;
        std::vector<size_t> aux_offsets;

        const std::vector<table_t>& total_sizes;
        std::vector<table_t> h_table;

        size_t num_chunks;

        std::vector<cudaEvent_t*> events;

        transfer_handler(const gpu_id_t num_gpus_,
                         const size_t num_phases_,
                         const std::vector<table_t>& total_sizes,
                         const size_t num_chunks_ = 1)
                         : num_gpus(num_gpus_),
                           num_phases(num_phases_),
                           phases(num_phases),
                           trg_offsets(num_gpus),
                           own_offsets(num_gpus),
                           aux_offsets(num_gpus),
                           total_sizes(total_sizes),
                        //    h_table(num_gpus+1),
                           num_chunks(num_chunks_)
        {
            for (gpu_id_t part = 0; part < num_gpus; ++part) {
                // horizontal scan to get initial offsets
                // h_table[part+1] = total_sizes[part]+h_table[part];
                // aux offsets begin at the end of own part
                aux_offsets[part] = total_sizes[part];
            }
        }

        ~transfer_handler() {
            for(auto& e : events)
                cudaEventDestroy(*e);
        }

        bool push_back(const std::vector<gpu_id_t>& sequence,
                       const size_t chunks = 1,
                       const bool verbose = false)
        {
            if (sequence.size() != num_phases+1)
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "sequence size does not match number of phases.");
                else return false;

            size_t* src_offset = &own_offsets[sequence.front()];
            const size_t size_per_chunk = SDIV(total_sizes[sequence.front()], num_chunks);
            size_t transfer_size = size_per_chunk * chunks;
            // check bounds
            if (*src_offset + transfer_size > total_sizes[sequence.front()])
                transfer_size = total_sizes[sequence.front()] - *src_offset;

            const gpu_id_t final_trg = sequence.back();

            size_t* trg_offset = nullptr;

            cudaEvent_t* event_before = nullptr;
            cudaEvent_t* event_after = nullptr;

            if (verbose)
                std::cout << "transfer from " << int(sequence.front())
                          << " to " << int(sequence.back())
                          << std::endl;

            if (sequence.front() == sequence.back()) { // src == trg
                // direct transfer (copy) in first phase
                trg_offset = &trg_offsets[sequence.back()];

                phases[0].emplace_back(sequence.front(), *src_offset,
                                       sequence.back(), *trg_offset,
                                       transfer_size,
                                       event_before, event_after);
                if (verbose) phases[0].back().show();

                // advance offsets
                *src_offset += transfer_size;
                *trg_offset += transfer_size;
            }
            else { // src != trg
                for (size_t phase = 0; phase < num_phases; ++phase) {
                    if (sequence[phase] != sequence[phase+1]) {
                        if (sequence[phase+1] != final_trg) {
                            // tranfer to auxiliary memory
                            trg_offset = &aux_offsets[sequence[phase+1]];
                            // create event after transfer for synchronization
                            event_after = new cudaEvent_t();
                            cudaSetDevice(sequence[phase]);
                            cudaEventCreate(event_after);
                            events.push_back(event_after);
                        }
                        else {
                            // tranfer to final memory position
                            trg_offset = &trg_offsets[sequence.front()];
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

    void show_phase(const std::vector<transfer>& transfers,
                    const std::vector<size_t  >& displacements) const {
        for(const transfer& t : transfers) {
            std::cout <<   "src:" << int(t.src_gpu)
                      << ", pos:" << t.src_pos
                      << ", trg:" << int(t.trg_gpu)
                      << ", pos:" << t.trg_pos
                      << ", len:" << t.len
                      << std::endl;
        }
    }

    template<typename index_t>
    bool check_recvbuf_size(const size_t buf_accessed,
                            const index_t buf_len) const {

        if (buf_accessed > buf_len)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "recvbuf not large enough for results.");
            else return false;
        return true;
    }

    template<typename index_t>
    bool check_transfers_size(const std::vector<size_t >& own_sizes,
                              const std::vector<size_t >& buffer_sizes,
                              const std::vector<index_t>& array_lens) const {

        for (gpu_id_t trg = 0; trg < num_gpus; trg++) {
            if (own_sizes[trg] > array_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "sendbuf access out of bounds.");
                else return false;

            if (buffer_sizes[trg] > array_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "sendbuf not large enough for buffer overhead.");
                else return false;
        }
        return true;
    }

    template<typename value_t>
    bool execute_phase(const std::vector<transfer>& transfers,
                       const std::vector<value_t *>& sendbufs,
                       const std::vector<value_t *>& recvbufs) const {

        if (sendbufs.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "recvbufs size does not match number of gpus.");
            else return false;

        if (recvbufs.size() != num_gpus)
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
            const value_t * from = sendbufs[t.src_gpu] + t.src_pos;
            value_t * to   = (t.trg_gpu == transfer_plan.get_main_gpu()) ?
                             recvbufs[t.src_gpu] + t.trg_pos :
                             sendbufs[t.trg_gpu] + t.trg_pos;

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
        const std::vector<value_t *>& sendbufs,       // sendbufs[k] resides on device_ids[k]
        const std::vector<index_t  >& sendbufs_lens,  // sendbufs_lens[k] is length of sendbufs[k]
        const std::vector<table_t  >& sendsizes,      // send sendsizes[k] bytes to device_ids[main_gpu]
        value_t * recvbuf,                            // recvbuf resides on device_ids[main_gpu]
        const index_t recvbuf_len                     // recvbuf_len is length of recvbuf
    ) const {

        if (!plan_valid) return false;

        if (sendbufs.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "sendbufs size does not match number of gpus.");
            else return false;
        if (sendbufs_lens.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "sendbufs_lens size does not match number of gpus.");
            else return false;
        if (sendsizes.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "sendsizes size does not match number of gpus.");
            else return false;

        const auto num_phases = transfer_plan.get_num_steps();
        const auto num_chunks = transfer_plan.get_num_chunks();

        transfer_handler<table_t> transfers(num_gpus, num_phases, sendsizes, num_chunks);

        // prepare transfers according to transfer_plan
        if (num_chunks > 1) {
            for (size_t i = 0; i < transfer_plan.get_transfer_sequences().size(); ++i) {
                transfers.push_back(transfer_plan.get_transfer_sequences()[i],
                                    transfer_plan.get_transfer_sizes()[i],
                                    true);
            }
        }
        else {
             for (const auto& sequence : transfer_plan.get_transfer_sequences()) {
                transfers.push_back(sequence, 1, true);
            }
        }

        std::vector<size_t> displacements(num_gpus+1);
        for (gpu_id_t part = 0; part < num_gpus; ++part) {
            // horizontal scan to get displacements
            displacements[part+1] = sendsizes[part] + displacements[part];
        }

        // for (size_t p = 0; p < num_phases; ++p)
        //     show_phase(transfers.phases[p], displacements);

        if(!check_recvbuf_size(displacements.back(), recvbuf_len))
            return false;
        if(!check_transfers_size(transfers.own_offsets, transfers.aux_offsets, sendbufs_lens))
            return false;

        // syncs with zero stream in order to enforce sequential
        // consistency with traditional synchronous memcpy calls
        if (!external_context)
            context->sync_hard();

        std::vector<value_t *> recvbufs(sendsizes.size());
        for (size_t i = 0; i < recvbufs.size(); ++i) {
            recvbufs[i] = recvbuf + displacements[i];
        }

        for (size_t p = 0; p < num_phases; ++p) {
            execute_phase(transfers.phases[p], sendbufs, recvbufs);
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

} // namespace

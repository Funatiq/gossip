#pragma once

#include <vector>

#include "config.h"
#include "common.cuh"
#include "context.cuh"
#include "scatter_plan.hpp"

namespace gossip {

class scatter_t {

private:
    const context_t * context;
    bool external_context;

    transfer_plan_t transfer_plan;
    bool plan_valid;

public:
    scatter_t (
        const gpu_id_t num_gpus_,
        const gpu_id_t main_gpu_)
        : context( new context_t(num_gpus_) ),
          external_context (false),
          transfer_plan( scatter::default_plan(num_gpus_, main_gpu_) ),
          plan_valid( transfer_plan.valid() )
    {}

    scatter_t (
        const gpu_id_t num_gpus_,
        const transfer_plan_t& transfer_plan_)
        : context( new context_t(num_gpus_) ),
          external_context(false),
          transfer_plan(transfer_plan_),
          plan_valid(false)
    {
        if(!transfer_plan.valid())
            scatter::verify_plan(transfer_plan);

        plan_valid = (get_num_devices() == transfer_plan.num_gpus()) &&
                     transfer_plan.valid();
    }

    scatter_t (
        const std::vector<gpu_id_t>& device_ids_,
        const gpu_id_t main_gpu_)
        : context( new context_t(device_ids_) ),
          external_context (false),
          transfer_plan( scatter::default_plan(device_ids_.size(), main_gpu_) ),
          plan_valid( transfer_plan.valid() )
    {}

    scatter_t (
        const std::vector<gpu_id_t>& device_ids_,
        const transfer_plan_t& transfer_plan_)
        : context( new context_t(device_ids_) ),
          external_context (false),
          transfer_plan(transfer_plan_),
          plan_valid(false)
    {
        if(!transfer_plan.valid())
            scatter::verify_plan(transfer_plan);

        plan_valid = (get_num_devices() == transfer_plan.num_gpus()) &&
                     transfer_plan.valid();
    }

     scatter_t (
        const context_t * context_,
        const gpu_id_t main_gpu_)
        : context(context_),
          external_context (true),
          transfer_plan( scatter::default_plan(context->get_num_devices(), main_gpu_) ),
          plan_valid( transfer_plan.valid() )
    {
        check(context->is_valid(),
              "You have to pass a valid context!");
    }

    scatter_t (
        const context_t * context_,
        const transfer_plan_t& transfer_plan_)
        : context(context_),
          external_context (true),
          transfer_plan(transfer_plan_),
          plan_valid(false)
    {
        check(context->is_valid(),
              "You have to pass a valid context!");

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
    template<typename table_t>
    struct transfer_handler {
        const context_t * context;

        std::vector<size_t> src_offsets;
        std::vector<size_t> trg_offsets;
        std::vector<size_t> aux_offsets;

        const std::vector<size_t>& src_displacements;
        const std::vector<table_t>& total_sizes;

        size_t num_phases;
        std::vector<std::vector<transfer> > phases;

        size_t num_chunks;

        std::vector<cudaEvent_t*> events;

        transfer_handler(
            const context_t * context_,
            const size_t num_phases_,
            const std::vector<size_t>& src_displacements,
            const std::vector<table_t>& total_sizes,
            const size_t num_chunks_ = 1
        ) :
            context(context_),
            // src offsets begin at src displacements
            src_offsets(src_displacements),
            trg_offsets(context->get_num_devices()),
            aux_offsets(context->get_num_devices()),
            src_displacements(src_displacements),
            total_sizes(total_sizes),
            num_phases(num_phases_),
            phases(num_phases),
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
            if(!check(sequence.size() == num_phases+1,
                      "sequence size does not match number of phases."))
                return false;

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

    template<typename value_t>
    bool execute_phase(
        const std::vector<transfer>& transfers,
        const value_t * srcs,
        const std::vector<value_t *>& dsts,
        const std::vector<value_t *>& bufs
    ) const {
        if (!check(dsts.size() == get_num_devices(),
                   "dsts size does not match number of gpus."))
            return false;
        if (!check(bufs.size() == get_num_devices(),
                    "dsts size does not match number of gpus."))
            return false;

        for(const transfer& t : transfers) {
            const gpu_id_t src = context->get_device_id(t.src_gpu);
            const gpu_id_t trg = context->get_device_id(t.trg_gpu);
            const auto stream  = context->get_streams(t.src_gpu)[t.trg_gpu];
            cudaSetDevice(src);
            const size_t size = t.len * sizeof(value_t);
            const value_t * from = (t.event_before == nullptr) ?
                             srcs + t.src_pos :
                             bufs[t.src_gpu] + t.src_pos;
            value_t * to   = (t.event_after == nullptr) ?
                             dsts[t.trg_gpu] + t.trg_pos :
                             bufs[t.trg_gpu] + t.trg_pos;

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
        const value_t * src,                     // src resides on device_ids[main_gpu]
        const index_t src_len,                   // src_len is length of src
        const std::vector<table_t  >& send_counts,     // send send_counts[k] elements to device_ids[k]
        const std::vector<value_t *>& dsts,      // dsts[k] resides on device_ids[k]
        const std::vector<index_t  >& dsts_lens  // dsts_len[k] is length of dsts[k]
    ) const {

        if (!plan_valid) return false;

        if (!check(dsts.size() == get_num_devices(),
                    "dsts size does not match number of gpus."))
            return false;
        if (!check(dsts_lens.size() == get_num_devices(),
                    "dsts_lens size does not match number of gpus."))
            return false;
        if (!check(send_counts.size() == get_num_devices(),
                    "table size does not match number of gpus."))
            return false;

        const auto num_phases = transfer_plan.num_steps();
        const auto num_chunks = transfer_plan.num_chunks();

        std::vector<size_t> displacements(get_num_devices()+1);
        for (gpu_id_t part = 0; part < get_num_devices(); ++part) {
            // exclusive scan to get displacements
            displacements[part+1] = send_counts[part] + displacements[part];
        }

        transfer_handler<table_t> transfers(context, num_phases,
                                            displacements, send_counts,
                                            num_chunks);

        bool verbose = false;
        // prepare transfers according to transfer_plan
        for (const auto& sequence : transfer_plan.transfer_sequences()) {
            transfers.push_back(sequence.seq, sequence.size, verbose);
        }

        // for (size_t p = 0; p < num_phases; ++p)
        //     show_transfers(transfers.phases[p]);

        if (!check_size(displacements.back(), src_len))
            return false;
        if(!check_size(transfers.trg_offsets, dsts_lens))
            return false;
        std::vector<size_t> total_offsets(transfers.trg_offsets);
        for (gpu_id_t i = 0; i < get_num_devices(); ++i)
            total_offsets[i] += transfers.aux_offsets[i];
        if(!check_size(total_offsets, dsts_lens))
            return false;

        // syncs with zero stream in order to enforce sequential
        // consistency with traditional synchronous memcpy calls
        if (!external_context)
            context->sync_hard();

        std::vector<value_t *> bufs(dsts);
        for (gpu_id_t i = 0; i < get_num_devices(); ++i)
            bufs[i] += send_counts[i];

        for (size_t p = 0; p < num_phases; ++p) {
            execute_phase(transfers.phases[p], src, dsts, bufs);
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

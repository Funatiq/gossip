#pragma once

#include <vector>

#include "config.h"
#include "common.cuh"
#include "context.cuh"
#include "all_to_all_plan.hpp"

namespace gossip {

class all2all_async_t {

    const context_t * context;

    transfer_plan_t transfer_plan;
    bool plan_valid;

public:
     all2all_async_t (
        const context_t& context_)
        : context(&context_),
          transfer_plan( all2all::default_plan(context->get_num_devices()) ),
          plan_valid( transfer_plan.valid() )
    {
        check(context->is_valid(),
              "You have to pass a valid context!");
    }

    all2all_async_t (
        const context_t& context_,
        const transfer_plan_t& transfer_plan_)
        : context(&context_),
          transfer_plan(transfer_plan_),
          plan_valid(false)
    {
        check(context->is_valid(),
              "You have to pass a valid context!");

        if(!transfer_plan.valid())
            all2all::verify_plan(transfer_plan);

        plan_valid = (get_num_devices() == transfer_plan.num_gpus()) &&
                     transfer_plan.valid();
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

        const std::vector<std::vector<size_t> >& src_displacements;
        std::vector<std::vector<size_t> > src_offsets;
        std::vector<std::vector<size_t> > trg_offsets;
        std::vector<size_t> aux_offsets;
        const std::vector<std::vector<table_t> >& sizes;

        size_t num_phases;
        std::vector<std::vector<transfer> > phases;

        size_t num_chunks;

        std::vector<cudaEvent_t*> events;

        transfer_handler(
            const context_t * context_,
            const std::vector<std::vector<size_t>>& src_displacements,
            const std::vector<std::vector<size_t>>& trg_displacements,
            const std::vector<std::vector<table_t>>& sizes,
            const size_t num_phases_,
            const size_t num_chunks_ = 1
        ) :
            context(context_),
            src_displacements(src_displacements),
            src_offsets(src_displacements),     // src offsets begin at src displacements
            trg_offsets(trg_displacements),     // trg offsets begin at trg displacements
            aux_offsets(context->get_num_devices()),
            sizes(sizes),
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
            const bool verbose = false
        ) {
            if(!check(sequence.size() == num_phases+1,
                      "sequence size does not match number of phases."))
                return false;

            size_t* src_offset = &src_offsets[sequence.front()][sequence.back()];
            const size_t size_per_chunk = SDIV(sizes[sequence.front()][sequence.back()], num_chunks);
            size_t transfer_size = size_per_chunk * chunks;
            // check bounds
            const size_t limit = src_displacements[sequence.front()][sequence.back()]
                               + sizes[sequence.front()][sequence.back()];
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
                const gpu_id_t src = sequence.front();
                const gpu_id_t trg = sequence.back();

                trg_offset = &trg_offsets[sequence.front()][sequence.back()];

                phases[phase].emplace_back(src, *src_offset,
                                           trg, *trg_offset,
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
                    const gpu_id_t src = sequence[phase];
                    const gpu_id_t trg = sequence[phase+1];
                    // schedule transfer only if device changes
                    if (src != trg) {
                        if (trg != final_trg) {
                            // transfer to auxiliary memory
                            trg_offset = &aux_offsets[trg];
                            // create event after transfer for synchronization
                            event_after = new cudaEvent_t();
                            const gpu_id_t id = context->get_device_id(src);
                            cudaSetDevice(id);
                            cudaEventCreate(event_after);
                            events.push_back(event_after);
                        }
                        else {
                            // transfer to final memory position
                            trg_offset = &trg_offsets[sequence.front()][sequence.back()];
                            // final transfer does not need follow up event
                            event_after = nullptr;
                        }

                        phases[phase].emplace_back(src, *src_offset,
                                                   trg, *trg_offset,
                                                   transfer_size,
                                                   event_before, event_after);
                        if (verbose) phases[phase].back().show();

                        // advance offset
                        *src_offset += transfer_size;
                        // old target is new source
                        src_offset = trg_offset;
                        event_before = event_after;

                        if (trg == final_trg)
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
        const std::vector<value_t *>& srcs,
        const std::vector<value_t *>& dsts,
        const std::vector<value_t *>& bufs
    ) const {
        if (!check(srcs.size() == get_num_devices(),
                   "srcs size does not match number of gpus."))
            return false;
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
            value_t * from = (t.event_before == nullptr) ?
                             srcs[t.src_gpu] + t.src_pos :
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
        std::vector<value_t *>& srcs,                   // src[k] resides on device_ids[k]
        const std::vector<index_t  >& srcs_lens,        // src_len[k] is length of src[k]
        std::vector<value_t *>& dsts,                   // dst[k] resides on device_ids[k]
        const std::vector<index_t  >& dsts_lens,        // dst_len[k] is length of dst[k]
        std::vector<value_t *>& bufs,
        const std::vector<index_t  >& bufs_lens,
        const std::vector<std::vector<table_t> >& send_counts // [src_gpu, partition]
    ) const {
        if (!plan_valid) return false;

        if (!check(srcs.size() == get_num_devices(),
                    "srcs size does not match number of gpus."))
            return false;
        if (!check(srcs_lens.size() == get_num_devices(),
                    "srcs_lens size does not match number of gpus."))
            return false;
        if (!check(dsts.size() == get_num_devices(),
                    "dsts size does not match number of gpus."))
            return false;
        if (!check(dsts_lens.size() == get_num_devices(),
                    "dsts_lens size does not match number of gpus."))
            return false;
        if (!check(send_counts.size() == get_num_devices(),
                    "table size does not match number of gpus."))
            return false;
        for (const auto& counts : send_counts) {
            if (!check(counts.size() == get_num_devices(),
                        "table size does not match number of gpus."))
                return false;
        }

        const auto num_phases = transfer_plan.num_steps();
        const auto num_chunks = transfer_plan.num_chunks();

        std::vector<std::vector<size_t> > src_displacements(get_num_devices(), std::vector<size_t>(get_num_devices()+1));
        // horizontal scan to get src offsets
        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu) {
            for (gpu_id_t part = 0; part < get_num_devices(); ++part) {
                src_displacements[gpu][part+1] = send_counts[gpu][part]+src_displacements[gpu][part];
            }
        }
        std::vector<std::vector<size_t> > trg_displacements(get_num_devices()+1, std::vector<size_t>(get_num_devices()));
        // vertical scan to get trg offsets
        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu) {
            for (gpu_id_t part = 0; part < get_num_devices(); ++part) {
                trg_displacements[part+1][gpu] = send_counts[part][gpu]+trg_displacements[part][gpu];
            }
        }

        transfer_handler<table_t> transfers(context,
                                            src_displacements,
                                            trg_displacements,
                                            send_counts,
                                            num_phases, num_chunks);

        bool verbose = false;
        // prepare transfers according to transfer_plan
        for (const auto& sequence : transfer_plan.transfer_sequences()) {
            transfers.push_back(sequence.seq, sequence.size, verbose);
        }

        // for (size_t p = 0; p < num_phases; ++p) {
        //     show_transfers(transfers.phases[p]);
        // }
        if(!check_size(transfers.aux_offsets, bufs_lens)) return false;
        if(!check_size(transfers.trg_offsets.back(), dsts_lens)) return false;

        for (size_t p = 0; p < num_phases; ++p) {
            execute_phase(transfers.phases[p], srcs, dsts, bufs);
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

#pragma once

#include <vector>
#include <stdexcept>

#include "config.h"
#include "context.cuh"
#include "all_to_all_plan.hpp"

namespace gossip {

template<
    bool throw_exceptions=true>
class all2all_async_t {

protected:
    const context_t<> * context;
private:
    bool external_context;

    const all2all_plan_t<> transfer_plan;

    bool plan_valid;

public:
    all2all_async_t (
        const gpu_id_t num_gpus_)
        : external_context (false),
          transfer_plan(num_gpus_)
    {
        context = new context_t<>(num_gpus_);

        plan_valid = transfer_plan.is_valid();
    }

    all2all_async_t (
        const gpu_id_t num_gpus_,
        const all2all_plan_t<>& transfer_plan_)
        : external_context (false),
          transfer_plan(transfer_plan_)
    {
        context = new context_t<>(num_gpus_);

        plan_valid = (get_num_devices() == transfer_plan.get_num_gpus()) &&
                     transfer_plan.is_valid();
    }

    all2all_async_t (
        const std::vector<gpu_id_t>& device_ids_)
        : external_context (false),
          transfer_plan(device_ids_.size())
    {
        context = new context_t<>(device_ids_);

        plan_valid = transfer_plan.is_valid();
    }

    all2all_async_t (
        const std::vector<gpu_id_t>& device_ids_,
        const all2all_plan_t<>& transfer_plan_)
        : external_context (false),
          transfer_plan(transfer_plan_)
    {
        context = new context_t<>(device_ids_);

        plan_valid = (get_num_devices() == transfer_plan.get_num_gpus()) &&
                     transfer_plan.is_valid();
    }

     all2all_async_t (
        const context_t<> * context_)
        : context(context_),
          external_context (true),
          transfer_plan(get_num_devices())
    {
        if (throw_exceptions)
            if (!context->is_valid())
                throw std::invalid_argument(
                    "You have to pass a valid context!"
                );

        plan_valid = transfer_plan.is_valid();
    }

    all2all_async_t (
        const context_t<> * context_,
        const all2all_plan_t<>& transfer_plan_)
        : context(context_),
          external_context (true),
          transfer_plan(transfer_plan_)
    {
        if (throw_exceptions)
            if (!context->is_valid())
                throw std::invalid_argument(
                    "You have to pass a valid context!"
                );

        plan_valid = (get_num_devices() == transfer_plan.get_num_gpus()) &&
                     transfer_plan.is_valid();
    }

    ~all2all_async_t () {
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
        const context_t<> * context;

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
            const context_t<> * context_,
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

        bool push_back(
            const std::vector<gpu_id_t>& sequence,
            const size_t chunks = 1,
            const bool verbose = false
        ) {
            if (sequence.size() != num_phases+1)
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "sequence size does not match number of phases.");
                else return false;

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
                    // schedule tranfer only if device changes
                    if (src != trg) {
                        if (trg != final_trg) {
                            // tranfer to auxiliary memory
                            trg_offset = &aux_offsets[trg];
                            // create event after transfer for synchronization
                            event_after = new cudaEvent_t();
                            const gpu_id_t id = context->get_device_id(src);
                            cudaSetDevice(id);
                            cudaEventCreate(event_after);
                            events.push_back(event_after);
                        }
                        else {
                            // tranfer to final memory position
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

    void show_phase(const std::vector<transfer>& transfers) const {
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
    bool check_phase_size(
        const std::vector<size_t >& transfer_sizes,
        const std::vector<index_t>& array_lens
    ) const {
        for (gpu_id_t trg = 0; trg < get_num_devices(); trg++) {
            if (transfer_sizes[trg] > array_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "array lens not compatible with transfer sizes.");
                else return false;
        }
        return true;
    }

    template<typename index_t>
    bool check_phase_size(
        const std::vector<std::vector<size_t > >& transfer_sizes,
        const std::vector<index_t>& array_lens
    ) const {
        for (gpu_id_t trg = 0; trg < get_num_devices(); trg++) {
            if (transfer_sizes[trg].back() > array_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "array lens not compatible with transfer sizes.");
                else return false;
        }
        return true;
    }

    template<typename value_t>
    bool execute_phase(
        const std::vector<transfer>& transfers,
        const std::vector<value_t *>& srcs,
        const std::vector<value_t *>& dsts,
        const std::vector<value_t *>& bufs
    ) const {
        if (srcs.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "srcs size does not match number of gpus.");
            else return false;
        if (dsts.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "dsts size does not match number of gpus.");
            else return false;
        if (bufs.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "dsts size does not match number of gpus.");
            else return false;

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

    // only for convenience
    template <
        typename value_t,
        typename index_t>
    bool clear(
        const std::vector<value_t *>& mem,
        const std::vector<index_t  >& mem_lens
    ) const {
        if (mem.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "mem size does not match number of gpus.");
            else return false;
        if (mem_lens.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "mem_lens size does not match number of gpus.");
            else return false;

        context->sync_all_streams();
        for (gpu_id_t gpu = 0; gpu < get_num_devices(); gpu++) {
            const gpu_id_t id = context->get_device_id(gpu);
            const auto stream = context->get_streams(gpu)[0];
            cudaSetDevice(id);
            const size_t size = mem_lens[gpu]
                              * sizeof(value_t);
            cudaMemsetAsync(mem[gpu], 0, size, stream);
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
        const std::vector<std::vector<table_t> >& sizes // [src_gpu, partition]
    ) const {
        if (!plan_valid) return false;

        if (srcs.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "srcs size does not match number of gpus.");
            else return false;
        if (srcs_lens.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "srcs_lens size does not match number of gpus.");
            else return false;
        if (dsts.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "dsts size does not match number of gpus.");
            else return false;
        if (dsts_lens.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "dsts_lens size does not match number of gpus.");
            else return false;
        if (sizes.size() != get_num_devices())
            if (throw_exceptions)
                throw std::invalid_argument(
                    "table size does not match number of gpus.");
            else return false;
        for (const auto& t : sizes)
            if (t.size() != get_num_devices())
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "table size does not match number of gpus.");
                else return false;

        const auto num_phases = transfer_plan.get_num_steps();
        const auto num_chunks = transfer_plan.get_num_chunks();

        std::vector<std::vector<size_t> > src_displacements(get_num_devices(), std::vector<size_t>(get_num_devices()+1));
        // horizontal scan to get src offsets
        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu) {
            for (gpu_id_t part = 0; part < get_num_devices(); ++part) {
                src_displacements[gpu][part+1] = sizes[gpu][part]+src_displacements[gpu][part];
            }
        }
        std::vector<std::vector<size_t> > trg_displacements(get_num_devices()+1, std::vector<size_t>(get_num_devices()));
        // vertical scan to get trg offsets
        for (gpu_id_t gpu = 0; gpu < get_num_devices(); ++gpu) {
            for (gpu_id_t part = 0; part < get_num_devices(); ++part) {
                trg_displacements[part+1][gpu] = sizes[part][gpu]+trg_displacements[part][gpu];
            }
        }

        transfer_handler<table_t> transfers(context,
                                            src_displacements,
                                            trg_displacements,
                                            sizes,
                                            num_phases, num_chunks);

        bool verbose = false;
        // prepare transfers according to transfer_plan
        if (num_chunks > 1) {
            for (size_t i = 0; i < transfer_plan.get_transfer_sequences().size(); ++i) {
                transfers.push_back(transfer_plan.get_transfer_sequences()[i],
                                    transfer_plan.get_transfer_sizes()[i],
                                    verbose);
            }
        }
        else {
             for (const auto& sequence : transfer_plan.get_transfer_sequences()) {
                transfers.push_back(sequence, 1, verbose);
            }
        }

        // for (size_t p = 0; p < num_phases; ++p) {
        //     show_phase(transfers.phases[p]);
        // }
        if(!check_phase_size(transfers.aux_offsets, bufs_lens)) return false;
        if(!check_phase_size(transfers.trg_offsets.back(), dsts_lens)) return false;

        // syncs with zero stream in order to enforce sequential
        // consistency with traditional synchronous memcpy calls
        if (!external_context)
            context->sync_hard();

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

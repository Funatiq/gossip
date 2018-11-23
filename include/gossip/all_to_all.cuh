#pragma once

#include <vector>

#include "config.h"
#include "all_to_all_plan.hpp"

namespace gossip {

template<
    bool throw_exceptions=true>
class all2all_t {

protected:
    context_t<> * context;
    gpu_id_t num_gpus;
private:
    bool external_context;

    const all2all_plan_t<> transfer_plan;

    bool plan_valid;

public:
    all2all_t (
        const gpu_id_t num_gpus_)
        : external_context (false),
          transfer_plan(num_gpus_)
    {
        context = new context_t<>(num_gpus_);
        num_gpus = context->get_num_devices();

        plan_valid = transfer_plan.is_valid();
    }

    all2all_t (
        const gpu_id_t num_gpus_,
        const all2all_plan_t<>& transfer_plan_)
        : external_context (false),
          transfer_plan(transfer_plan_)
    {
        context = new context_t<>(num_gpus_);
        num_gpus = context->get_num_devices();

        plan_valid = (num_gpus == transfer_plan.get_num_gpus()) &&
                     transfer_plan.is_valid();
    }

    all2all_t (
        const std::vector<gpu_id_t>& device_ids_)
        : external_context (false),
          transfer_plan(device_ids_.size())
    {
        context = new context_t<>(device_ids_);
        num_gpus = context->get_num_devices();

        plan_valid = transfer_plan.is_valid();
    }

    all2all_t (
        const std::vector<gpu_id_t>& device_ids_,
        const all2all_plan_t<>& transfer_plan_)
        : external_context (false),
          transfer_plan(transfer_plan_)
    {
        context = new context_t<>(device_ids_);
        num_gpus = context->get_num_devices();

        plan_valid = (num_gpus == transfer_plan.get_num_gpus()) &&
                     transfer_plan.is_valid();
    }

     all2all_t (
        context_t<> * context_)
        : context(context_),
          num_gpus(context->get_num_devices()),
          external_context (true),
          transfer_plan(num_gpus)
    {
        if (throw_exceptions)
            if (!context->is_valid())
                throw std::invalid_argument(
                    "You have to pass a valid context!"
                );

        plan_valid = transfer_plan.is_valid();
    }

    all2all_t (
        context_t<> * context_,
        const all2all_plan_t<>& transfer_plan_)
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

    ~all2all_t () {
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
        gpu_id_t num_gpus;
        size_t num_phases;
        std::vector<std::vector<transfer> > phases;
        std::vector<std::vector<size_t> > phases_offsets;

        const std::vector<std::vector<table_t> >& table;
        std::vector<std::vector<table_t> > h_table;

        size_t num_chunks;

        transfer_handler(const gpu_id_t num_gpus_,
                         const size_t num_phases_,
                         const std::vector<std::vector<table_t>>& table,
                         const size_t num_chunks_ = 1)
                         : num_gpus(num_gpus_),
                           num_phases(num_phases_),
                           phases(num_phases),
                           phases_offsets(num_phases, std::vector<size_t>(num_gpus)),
                           table(table),
                           h_table(num_gpus, std::vector<table_t>(num_gpus+1)),
                           num_chunks(num_chunks_) {

            // horizontal scan to get initial offsets
            for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
                for (gpu_id_t part = 0; part < num_gpus; ++part) {
                    h_table[gpu][part+1] = table[gpu][part]+h_table[gpu][part];
                }
            }
        }

        bool push_back(const std::vector<gpu_id_t>& sequence,
                       const size_t chunks = 1) {

            if (sequence.size() != num_phases+1)
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "sequence size does not match number of phases.");
                else return false;

            const size_t offset = h_table[sequence.front()][sequence.back()];
            const size_t size_per_chunk = SDIV(table[sequence.front()][sequence.back()], num_chunks);
            size_t transfer_size = size_per_chunk * chunks;
            if (offset + transfer_size > h_table[sequence.front()][sequence.back()+1])
                transfer_size = h_table[sequence.front()][sequence.back()+1] - offset;

            size_t phase = 0;
            phases[phase].emplace_back(sequence[phase], offset,
                                       sequence[phase+1], phases_offsets[phase][sequence[phase+1]],
                                       transfer_size);
            h_table[sequence.front()][sequence.back()] += transfer_size;

            for (size_t phase = 1; phase < num_phases; ++phase) {
                phases[phase].emplace_back(sequence[phase], phases_offsets[phase-1][sequence[phase]],
                                           sequence[phase+1], phases_offsets[phase][sequence[phase+1]],
                                           transfer_size);
            }
            for (size_t phase = 0; phase < num_phases; ++phase) {
                phases_offsets[phase][sequence[phase+1]] += transfer_size;
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
    bool check_phase_size(const std::vector<size_t >& transfer_sizes,
                          const std::vector<index_t>& array_lens) const {
        for (gpu_id_t trg = 0; trg < num_gpus; trg++) {
            if (transfer_sizes[trg] > array_lens[trg])
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "array lens not compatible with transfer sizes.");
                else return false;
        }
        return true;
    }

    template<typename value_t>
    bool execute_phase(const std::vector<transfer>& transfers,
                       const std::vector<value_t *>& srcs,
                       const std::vector<value_t *>& dsts) const {

        if (srcs.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "srcs size does not match number of gpus.");
            else return false;
        if (dsts.size() != num_gpus)
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
            value_t * from = srcs[t.src_gpu] + t.src_pos;
            value_t * to   = dsts[t.trg_gpu] + t.trg_pos;

            cudaMemcpyPeerAsync(to, trg, from, src, size, stream);
        } CUERR

        return true;
    }

    // only for convenience
    template <
        typename value_t,
        typename index_t>
    bool clear(const std::vector<value_t *>& mem,
               const std::vector<index_t  >& mem_lens) const {

        if (mem.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "mem size does not match number of gpus.");
            else return false;
        if (mem_lens.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "mem_lens size does not match number of gpus.");
            else return false;

        context->sync_all_streams();
        for (gpu_id_t gpu = 0; gpu < num_gpus; gpu++) {
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
        std::vector<value_t *>& srcs,      // src[k] resides on device_ids[k]
        const std::vector<index_t  >& srcs_lens, // src_len[k] is length of src[k]
        std::vector<value_t *>& dsts,      // dst[k] resides on device_ids[k]
        const std::vector<index_t  >& dsts_lens, // dst_len[k] is length of dst[k]
        const std::vector<std::vector<table_t> >& table) const {  // [src_gpu, partition]

        if (!plan_valid) return false;

        if (srcs.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "srcs size does not match number of gpus.");
            else return false;
        if (srcs_lens.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "srcs_lens size does not match number of gpus.");
            else return false;
        if (dsts.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "dsts size does not match number of gpus.");
            else return false;
        if (dsts_lens.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "dsts_lens size does not match number of gpus.");
            else return false;
        if (table.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "table size does not match number of gpus.");
            else return false;
        for (const auto& t : table)
            if (t.size() != num_gpus)
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "table size does not match number of gpus.");
                else return false;

        const auto num_phases = transfer_plan.get_num_steps();
        const auto num_chunks = transfer_plan.get_num_chunks();

        transfer_handler<table_t> transfers(num_gpus, num_phases, table, num_chunks);

        // prepare transfers according to transfer_plan
        if (num_chunks > 1) {
            for (size_t i = 0; i < transfer_plan.get_transfer_sequences().size(); ++i) {
                transfers.push_back(transfer_plan.get_transfer_sequences()[i],
                                    transfer_plan.get_transfer_sizes()[i]);
            }
        }
        else {
             for (const auto& sequence : transfer_plan.get_transfer_sequences()) {
                transfers.push_back(sequence);
            }
        }

        for (size_t p = 0; p < num_phases; ++p) {
            show_phase(transfers.phases[p]);
            if(!check_phase_size(transfers.phases_offsets[p], dsts_lens)) return false;
        }

        // syncs with zero stream in order to enforce sequential
        // consistency with traditional synchronous memcpy calls
        if (!external_context)
            context->sync_hard();

        for (size_t p = 0; p < num_phases; ++p) {
            execute_phase(transfers.phases[p], srcs, dsts);

            if (p < num_phases-1) {
                // swap srcs and dsts for next phase
                srcs.swap(dsts);

                // mandatory sync between phases
                context->sync_all_streams();
            }
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

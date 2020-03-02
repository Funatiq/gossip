#pragma once

#include <vector>

#include "config.h"
#include "error_checking.hpp"
#include "context.cuh"
#include "common.cuh"
#include "all_to_all_plan.hpp"

namespace gossip {

class all2all_t {

    const context_t * context;

    transfer_plan_t transfer_plan;
    bool plan_valid;

public:
    all2all_t (
        const context_t& context_)
        : context(&context_),
          transfer_plan( all2all::default_plan(context->get_num_devices()) ),
          plan_valid( transfer_plan.valid() )
    {
        check(context->is_valid(),
              "You have to pass a valid context!");
    }

    all2all_t (
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

        check(get_num_devices() == transfer_plan.num_gpus(),
              "Plan does fit number of gpus of context!");

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

        void show() const {
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
        const context_t * context;

        std::vector<std::vector<size_t> > src_offsets;
        std::vector<std::vector<size_t> > phases_offsets;
        std::vector<std::vector<size_t> > trg_offsets;

        const std::vector<std::vector<size_t> >& src_displacements;
        const std::vector<std::vector<table_t> >& sizes;

        size_t num_phases;
        std::vector<std::vector<transfer> > phases;

        size_t num_chunks;

        transfer_handler(
            const context_t * context_,
            const std::vector<std::vector<table_t>>& src_displacements,
            const std::vector<std::vector<table_t>>& trg_displacements,
            const std::vector<std::vector<table_t>>& sizes,
            const size_t num_phases_,
            const size_t num_chunks_ = 1
        ) :
            context(context_),
            src_offsets(src_displacements), // src offsets begin at src displacements
            phases_offsets(),
            trg_offsets(trg_displacements), // trg offsets begin at trg displacements
            src_displacements(src_displacements),
            sizes(sizes),
            num_phases(num_phases_),
            phases(num_phases),
            num_chunks(num_chunks_)
        {
            if(num_phases > 1)
                phases_offsets.resize(num_phases-1, std::vector<size_t>(context->get_num_devices()));
        }

        bool push_back(
            const std::vector<gpu_id_t>& sequence,
            const size_t chunks = 1
        ) {
            if(!check(sequence.size() == num_phases+1,
                      "sequence size does not match number of phases."))
                return false;

            const size_t size_per_chunk = SDIV(sizes[sequence.front()][sequence.back()], num_chunks);
            size_t transfer_size = size_per_chunk * chunks;

            const size_t src_offset = src_offsets[sequence.front()][sequence.back()];
            const size_t trg_offset = trg_offsets[sequence.front()][sequence.back()];
            // check bounds
            const size_t limit = src_displacements[sequence.front()][sequence.back()]
                               + sizes[sequence.front()][sequence.back()];
            if (src_offset + transfer_size > limit)
                transfer_size = limit - src_offset;

            if (num_phases == 1) {
                size_t phase = 0;
                phases[phase].emplace_back(sequence[phase], src_offset,
                                           sequence[phase+1], trg_offset,
                                           transfer_size);
            }
            else {
                size_t phase = 0;
                phases[phase].emplace_back(sequence[phase], src_offset,
                                           sequence[phase+1], phases_offsets[phase][sequence[phase+1]],
                                           transfer_size);

                for (phase = 1; phase < num_phases-1; ++phase) {
                    phases[phase].emplace_back(sequence[phase], phases_offsets[phase-1][sequence[phase]],
                                               sequence[phase+1], phases_offsets[phase][sequence[phase+1]],
                                               transfer_size);
                }

                phase = num_phases-1;
                phases[phase].emplace_back(sequence[phase], phases_offsets[phase-1][sequence[phase]],
                                           sequence[phase+1], trg_offset,
                                           transfer_size);
            }

            src_offsets[sequence.front()][sequence.back()] += transfer_size;
            for (size_t phase = 0; phase < num_phases-1; ++phase) {
                phases_offsets[phase][sequence[phase+1]] += transfer_size;
            }
            trg_offsets[sequence.front()][sequence.back()] += transfer_size;

            return true;
        }

        void show_phase(const size_t phase) const {
            for(const transfer& t : phases[phase]) {
                t.show();
            }
        }

        template<typename value_t>
        bool execute_phase(
            const size_t phase,
            const std::vector<value_t *>& srcs,
            const std::vector<value_t *>& dsts
        ) const {
            for(const transfer& t : phases[phase]) {
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
    };

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
        const std::vector<std::vector<table_t> >& send_counts, // [src_gpu, partition]
        bool verbose = false
    ) const {
        if (!check(plan_valid, "Invalid plan. Abort."))
            return false;

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
        // horizontal scan to get initial offsets
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

        // prepare transfers according to transfer_plan
        for (const auto& sequence : transfer_plan.transfer_sequences()) {
            transfers.push_back(sequence.seq, sequence.size);
        }

        if(verbose) {
            for (size_t p = 0; p < num_phases; ++p) {
                transfers.show_phase(p);
            }
        }
        for (size_t p = 0; p < num_phases-1; ++p) {
            if(!check_size(transfers.phases_offsets[p], dsts_lens)) return false;
        }
        if(!check_size(transfers.trg_offsets[get_num_devices()], dsts_lens)) return false;

        for (size_t p = 0; p < num_phases; ++p) {
            transfers.execute_phase(p, srcs, dsts);

            if (p < num_phases-1) {
                // swap srcs and dsts for next phase
                srcs.swap(dsts);

                // mandatory sync between phases
                context->sync_all_streams();
            }
        }

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

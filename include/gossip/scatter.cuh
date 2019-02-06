#pragma once

#include <vector>

#include "config.h"
#include "error_checking.hpp"
#include "common.cuh"
#include "context.cuh"
#include "scatter_plan.hpp"

namespace gossip {

class scatter_t {

private:
    const context_t * context;

    transfer_plan_t transfer_plan;
    bool plan_valid;

public:
     scatter_t (
        const context_t& context_,
        const gpu_id_t main_gpu_)
        : context(&context_),
          transfer_plan( scatter::default_plan(context->get_num_devices(), main_gpu_) ),
          plan_valid( transfer_plan.valid() )
    {
        check(context->is_valid(),
              "You have to pass a valid context!");
    }

    scatter_t (
        const context_t& context_,
        const transfer_plan_t& transfer_plan_)
        : context(&context_),
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

public:
    void show_plan() const {
        if(!plan_valid)
            std::cout << "WARNING: plan does fit number of gpus\n";

        transfer_plan.show_plan();
    }

public:
    template <
        typename value_t,
        typename index_t,
        typename table_t>
    bool execAsync (
        value_t * const src,                     // src resides on device_ids[main_gpu]
        const index_t src_len,                   // src_len is length of src
        const std::vector<table_t  >& send_counts,     // send send_counts[k] elements to device_ids[k]
        const std::vector<value_t *>& dsts,      // dsts[k] resides on device_ids[k]
        const std::vector<index_t  >& dsts_lens,  // dsts_len[k] is length of dsts[k]
        bool verbose = false
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

        const auto main_gpu   = transfer_plan.main_gpu();
        const auto num_phases = transfer_plan.num_steps();
        const auto num_chunks = transfer_plan.num_chunks();

        std::vector<std::vector<size_t> > src_displacements(get_num_devices(), std::vector<size_t>(get_num_devices()+1));
        for (gpu_id_t part = 0; part < get_num_devices(); ++part) {
            // exclusive scan to get src_displacements
            src_displacements[main_gpu][part+1] = send_counts[part] + src_displacements[main_gpu][part];
        }

        std::vector<std::vector<size_t> > trg_displacements(get_num_devices()+1, std::vector<size_t>(get_num_devices()));

        std::vector<std::vector<table_t> > sizes(get_num_devices(), std::vector<table_t>(get_num_devices()));
        for (gpu_id_t part = 0; part < get_num_devices(); ++part) {
            sizes[main_gpu][part] = send_counts[part];
        }

        transfer_handler<table_t> transfers(context,
                                            src_displacements,
                                            trg_displacements,
                                            sizes,
                                            num_phases, num_chunks);

        // prepare transfers according to transfer_plan
        for (const auto& sequence : transfer_plan.transfer_sequences()) {
            transfers.push_back(sequence.seq, sequence.size, verbose);
        }

        if(verbose) {
            for (size_t p = 0; p < num_phases; ++p) {
                transfers.show_phase(p);
            }
        }

        // check source array size
        if (!check_size(src_displacements[main_gpu].back(), src_len))
            return false;
        std::vector<value_t *> srcs(get_num_devices());
        srcs[transfer_plan.main_gpu()] = src;

        // buffer behind local destinations
        for (gpu_id_t i = 0; i < get_num_devices(); ++i)
            if(!check_size(transfers.trg_offsets[main_gpu][i] + transfers.aux_offsets[i], dsts_lens[i]))
                return false;
        std::vector<value_t *> bufs(dsts);
        for (gpu_id_t i = 0; i < get_num_devices(); ++i)
            bufs[i] += send_counts[i];

        for (size_t p = 0; p < num_phases; ++p) {
            transfers.execute_phase(p, srcs, dsts, bufs);
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

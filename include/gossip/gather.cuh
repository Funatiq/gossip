#pragma once

#include <vector>

#include "config.h"
#include "error_checking.hpp"
#include "common.cuh"
#include "context.cuh"
#include "gather_plan.hpp"

namespace gossip {

class gather_t {

private:
    const context_t * context;

    transfer_plan_t transfer_plan;
    bool plan_valid;

public:
     gather_t (
        const context_t& context_,
        const gpu_id_t main_gpu_)
        : context(&context_),
          transfer_plan( gather::default_plan(context->get_num_devices(), main_gpu_) ),
          plan_valid( transfer_plan.valid() )
    {
        check(context->is_valid(),
              "You have to pass a valid context!");
    }

    gather_t (
        const context_t& context_,
        const transfer_plan_t& transfer_plan_)
        : context(&context_),
          transfer_plan(transfer_plan_),
          plan_valid(false)
    {
        check(context->is_valid(),
              "You have to pass a valid context!");

        if(!transfer_plan.valid())
            gather::verify_plan(transfer_plan);

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
    template <
        typename table_t>
    transfer_handler<table_t> makeTransferHandler (
        const std::vector<table_t  >& send_counts,
        bool verbose = false
    ) const {
        const auto main_gpu   = transfer_plan.main_gpu();
        const auto num_phases = transfer_plan.num_steps();
        const auto num_chunks = transfer_plan.num_chunks();

        std::vector<std::vector<size_t> > src_displacements(get_num_devices(), std::vector<size_t>(get_num_devices()+1));

        std::vector<std::vector<size_t> > trg_displacements(get_num_devices()+1, std::vector<size_t>(get_num_devices()));
        for (gpu_id_t part = 0; part < get_num_devices(); ++part) {
            // exclusive scan to get trg_displacements
            trg_displacements[part+1][main_gpu] = send_counts[part] + trg_displacements[part][main_gpu];
        }

        std::vector<std::vector<table_t> > sizes(get_num_devices(), std::vector<table_t>(get_num_devices()));
        for (gpu_id_t part = 0; part < get_num_devices(); ++part) {
            sizes[part][main_gpu] = send_counts[part];
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

        return transfers;
    }

public:
    /**
     * Calculate buffer lengths needed to execute scatter with given send_counts.
     * The lenghts of the parameters have to match the context.
     * @param send_counts send_counts[k] elements are sent to device_ids[k].
     * @param verbose if true, show details for each transfer.
     * @return bufs_len bufs_len[k] is required length of bufs[k] array.
     */
    template <
        typename table_t>
    const std::vector<size_t> calcBufferLengths (
        const std::vector<table_t>& send_counts,
        bool verbose = false
    ) const {
        if (!check(plan_valid, "Invalid plan. Abort."))
            return {};

        if (!check(send_counts.size() == get_num_devices(),
                    "table size does not match number of gpus."))
            return {};

        transfer_handler<table_t> transfers = makeTransferHandler(send_counts, verbose);

        return transfers.aux_offsets;
    }

    /**
     * Execute gather asynchronously using the given context.
     * The lenghts of the parameters have to match the context.
     * @param srcs pointers to source arrays. srcs[k] array should reside on device_ids[k].
     * @param srcs_len srcs_len[k] is length of srcs[k] array.
     * @param send_counts send_counts[k] elements are sent from device_ids[k].
     * @param dst pointer to destination array. should reside on device_ids[main_gpu].
     * @param dst_len dst_len is length of dst array.
     * @param verbose if true, show details for each transfer.
     * @return true if executed successfully.
     */
    template <
        typename value_t,
        typename index_t,
        typename table_t>
    bool execAsync (
        const std::vector<value_t *>& srcs,
        const std::vector<index_t  >& srcs_lens,
        value_t * dst,
        const index_t dst_len,
        const std::vector<value_t *>& bufs,
        const std::vector<index_t  >& bufs_lens,
        const std::vector<table_t  >& send_counts,
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
        if (!check(bufs.size() == get_num_devices(),
                    "bufs size does not match number of gpus."))
            return false;
        if (!check(bufs_lens.size() == get_num_devices(),
                    "bufs_lens size does not match number of gpus."))
            return false;
        if (!check(send_counts.size() == get_num_devices(),
                    "table size does not match number of gpus."))
            return false;

        transfer_handler<table_t> transfers = makeTransferHandler(send_counts, verbose);

        // check source array sizes
        if (!check_size(send_counts, srcs_lens)) return false;
        // check buffer array sizes
        if(!check_size(transfers.aux_offsets, bufs_lens)) return false;
        // check destination array size
        if(!check_size(transfers.trg_offsets.back()[transfer_plan.main_gpu()], dst_len))
            return false;

        std::vector<value_t *> dsts(get_num_devices());
        dsts[transfer_plan.main_gpu()] = dst;

        for (size_t p = 0; p < transfers.num_phases; ++p)
            transfers.execute_phase(p, srcs, dsts, bufs);

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

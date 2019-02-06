#pragma once

#include <vector>
#include <iostream>

#include "error_checking.hpp"
#include "transfer_plan.hpp"

namespace gossip {

class gather {

public:
    static void verify_plan(transfer_plan_t& plan) {
        bool valid = true;

        valid &= check(plan.main_gpu() != gpu_id_t(-1),
                    "main gpu not set in plan.");

        valid &= check(plan.num_steps() >= 1,
                    "planned sequence must be at least of length 2.");

        for (const auto& sequence : plan.transfer_sequences())
            valid &= check(sequence.seq.size() == plan.num_steps()+1,
                        "planned sequences must have same lengths.");

        for (const auto& sequence : plan.transfer_sequences()) {
            valid &= check(sequence.seq.back() == plan.main_gpu(),
                        "all sequences must have same target.");
        }

        std::vector<size_t> completeness(plan.num_gpus());
        // sum up all chunks for each source gpu
        for (const auto& sequence : plan.transfer_sequences()) {
            completeness[sequence.seq.front()] += sequence.size;
        }
        for (gpu_id_t trg = 0; trg < plan.num_gpus(); ++trg) {
            valid &= check(completeness[trg] == plan.num_chunks(),
                        "transfer plan is incomplete.");
        }

        if(valid)
            plan.validate();
    }

    static transfer_plan_t default_plan(const gpu_id_t num_gpus, const gpu_id_t target) {

        std::vector<std::vector<gpu_id_t> > sequences;

        sequences.reserve(num_gpus);

        // plan direct transfers from src to target gpu
        for (gpu_id_t src = 0; src < num_gpus; ++src) {
            sequences.emplace_back(std::vector<gpu_id_t>{src,target});
        }

        transfer_plan_t plan(num_gpus, sequences);

        plan.main_gpu(target);

        verify_plan(plan);

        return plan;
    }

};

} // namespace

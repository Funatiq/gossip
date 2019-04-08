#pragma once

#include <vector>
#include <iostream>

#include "error_checking.hpp"
#include "transfer_plan.hpp"

namespace gossip {

class all2all {

public:
    static void verify_plan(transfer_plan_t& plan) {
        bool valid = true;

        valid &= check(plan.num_steps() >= 1,
                    "planned sequence must be at least of length 2.");

        for (const auto& sequence : plan.transfer_sequences())
            valid &= check(sequence.seq.size() == plan.num_steps()+1,
                        "planned sequences must have same lengths.");

        std::vector<std::vector<size_t> > completeness(plan.num_gpus(), std::vector<size_t>(plan.num_gpus()));
        for (const auto& sequence : plan.transfer_sequences()) {
            completeness[sequence.seq.front()][sequence.seq.back()] += sequence.size;
        }
        for (gpu_id_t src = 0; src < plan.num_gpus(); ++src) {
            for (gpu_id_t trg = 0; trg < plan.num_gpus(); ++trg) {
                valid &= check(completeness[src][trg] == plan.num_chunks(),
                            "transfer plan is incomplete.");
            }
        }

        if(valid)
            plan.validate();
    }

    static transfer_plan_t default_plan(const gpu_id_t num_gpus)  {

        std::vector<std::vector<gpu_id_t> > sequences;

        sequences.reserve(num_gpus*num_gpus);

        // plan direct transfers from src to trg gpu
        for (gpu_id_t src = 0; src < num_gpus; ++src) {
            for (gpu_id_t trg = 0; trg < num_gpus; ++trg) {
                sequences.emplace_back(std::vector<gpu_id_t>{src,trg});
            }
        }

        transfer_plan_t plan("all2all", num_gpus, sequences);

        verify_plan(plan);

        return plan;
    }

};

} // namespace

#pragma once

#include <vector>
#include <iostream>

#include "transfer_plan.hpp"

namespace gossip {

template<
    bool throw_exceptions=true>
class gather_plan_t : public transfer_plan_t<throw_exceptions> {

    gpu_id_t target;

public:
    gather_plan_t(const gpu_id_t target_,
                  const gpu_id_t num_gpus_,
                  const std::vector<std::vector<gpu_id_t>>& transfer_sequences_ = {})
                  : transfer_plan_t<throw_exceptions>(num_gpus_, transfer_sequences_),
                    target(target_)
    {
        if(this->transfer_sequences.empty()) load_default_plan();
        this->synchronized = false;
        this->valid = verify_plan();
    }

    gather_plan_t(const gpu_id_t target_,
                  const gpu_id_t num_gpus_,
                  const std::vector<std::vector<gpu_id_t>>& transfer_sequences_,
                  const size_t num_chunks_,
                  const std::vector<size_t>& transfer_sizes_)
                  : transfer_plan_t<throw_exceptions>(num_gpus_, transfer_sequences_,
                                                      num_chunks_, transfer_sizes_),
                     target(target_)
    {
        if(this->transfer_sequences.empty()) load_default_plan();
        this->synchronized = false;
        this->valid = verify_plan();
    }

private:
    void load_default_plan() override {
        this->num_steps = 1;
        this->num_chunks = 1;

        this->transfer_sequences.clear();
        this->transfer_sequences.reserve(this->num_gpus);

        // plan direct transfers from src to target gpu
        for (gpu_id_t src = 0; src < this->num_gpus; ++src) {
            this->transfer_sequences.emplace_back(std::vector<gpu_id_t>{src,target});
        }
    }

    bool verify_plan() const override {

        for (const auto& sequence : this->transfer_sequences) {
            if (sequence.back() != target)
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "all sequences must have same target.");
                else return false;
        }

        std::vector<size_t> completeness(this->num_gpus);
        if (this->num_chunks <= 1) {
            for (const auto& sequence : this->transfer_sequences) {
                completeness[sequence.front()] += 1;
            }
        }
        else {
            if (this->transfer_sequences.size() != this->transfer_sizes.size())
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "number of sequences must match number of sizes.");
                else return false;
           for (size_t i = 0; i < this->transfer_sequences.size(); ++i) {
                completeness[this->transfer_sequences[i].front()]
                    += this->transfer_sizes[i];
            }           
        }
        for (gpu_id_t src = 0; src < this->num_gpus; ++src) {
            if (completeness[src] != this->num_chunks)
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "transfer plan is incomplete.");
                else return false;
        }

        return true;
    }

};

} // namespace

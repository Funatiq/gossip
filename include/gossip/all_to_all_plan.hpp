#pragma once

#include <vector>
#include <iostream>

#include "transfer_plan.hpp"

namespace gossip {

template<
    bool throw_exceptions=true>
class all2all_plan_t : public transfer_plan_t<throw_exceptions> {

public:
    all2all_plan_t(const gpu_id_t num_gpus_,
                   const std::vector<std::vector<gpu_id_t>>& transfer_sequences_ = {})
                   : transfer_plan_t<throw_exceptions>(num_gpus_, transfer_sequences_)
    {
        if(this->transfer_sequences.empty()) load_default_plan();
        this->num_steps = this->transfer_sequences[0].size()-1;
        this->synchronized = true;
        this->valid = verify_plan();
    }

    all2all_plan_t(const gpu_id_t num_gpus_,
                   const std::vector<std::vector<gpu_id_t>>& transfer_sequences_,
                   const size_t num_chunks_,
                   const std::vector<size_t>& transfer_sizes_)
                   : transfer_plan_t<throw_exceptions>(num_gpus_, transfer_sequences_,
                                                       num_chunks_, transfer_sizes_)
    {
        if(this->transfer_sequences.empty()) load_default_plan();
        this->num_steps = this->transfer_sequences[0].size()-1;
        this->synchronized = true;
        this->valid = verify_plan();
    }

private:
    void load_default_plan() override {
        this->num_steps = 1;
        this->num_chunks = 1;

        this->transfer_sequences.clear();
        this->transfer_sequences.reserve(this->num_gpus*this->num_gpus);

        // plan direct transfers from src to trg gpu
        for (gpu_id_t src = 0; src < this->num_gpus; ++src) {
            for (gpu_id_t trg = 0; trg < this->num_gpus; ++trg) {
                this->transfer_sequences.emplace_back(std::vector<gpu_id_t>{src,trg});
            }
        }
    }

    bool verify_plan() const override {
        if (this->num_steps < 1)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "planned sequence must be at least of length 2.");
            else return false;

        for (const auto& sequence : this->transfer_sequences)
            if (sequence.size() != this->num_steps+1)
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "planned sequences must have same lengths.");
                else return false;

        std::vector<std::vector<size_t> > completeness(this->num_gpus, std::vector<size_t>(this->num_gpus));
        if (this->num_chunks <= 1) {
            for (const auto& sequence : this->transfer_sequences) {
                completeness[sequence.front()][sequence.back()] += 1;
            }
        }
        else {
            if (this->transfer_sequences.size() != this->transfer_sizes.size())
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "number of sequences must match number of sizes.");
                else return false;
           for (size_t i = 0; i < this->transfer_sequences.size(); ++i) {
                completeness[this->transfer_sequences[i].front()][this->transfer_sequences[i].back()]
                    += this->transfer_sizes[i];
            }           
        }
        for (gpu_id_t src = 0; src < this->num_gpus; ++src) {
            for (gpu_id_t trg = 0; trg < this->num_gpus; ++trg) {
                if (completeness[src][trg] != this->num_chunks)
                    if (throw_exceptions)
                        throw std::invalid_argument(
                            "transfer plan is incomplete.");
                    else return false;
            }
        }

        return true;
    }

};

} // namespace

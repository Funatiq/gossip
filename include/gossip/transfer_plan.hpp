#pragma once

#include <vector>
#include <iostream>

#include "config.h"

namespace gossip {

class transfer_plan_t {

    struct transfer_sequence {
        std::vector<gpu_id_t> seq;
        size_t size;
    };

    std::string type_;
    gpu_id_t num_gpus_;
    gpu_id_t main_gpu_;
    size_t num_steps_;
    size_t num_chunks_;
    std::vector<transfer_sequence> transfer_sequences_;
    std::vector<size_t> sync_steps_;
    bool valid_;

public:
    transfer_plan_t(
        const std::string type,
        const gpu_id_t num_gpus,
        const std::vector<std::vector<gpu_id_t>>& sequences
    ) :
        type_(type),
        num_gpus_(num_gpus),
        main_gpu_(gpu_id_t(-1)),
        num_steps_(0),
        num_chunks_(1),
        valid_(false)
    {
        if(sequences.size())
            num_steps_ = sequences[0].size()-1;
        transfer_sequences_.reserve(sequences.size());
        for(const auto& sequence : sequences)
            transfer_sequences_.push_back({sequence, 1});
    }

    transfer_plan_t(
        const std::string type,
        const gpu_id_t num_gpus,
        const std::vector<std::vector<gpu_id_t>>& sequences,
        const size_t num_chunks,
        const std::vector<size_t>& transfer_sizes
    ) :
        type_(type),
        num_gpus_(num_gpus),
        main_gpu_(gpu_id_t(-1)),
        num_steps_(0),
        num_chunks_(num_chunks),
        valid_(false)
    {
        if(sequences.size() == transfer_sizes.size()) {
            if(sequences.size())
                num_steps_ = sequences[0].size()-1;

            transfer_sequences_.reserve(sequences.size());

            for(size_t i = 0; i < sequences.size(); ++i)
                transfer_sequences_.push_back({sequences[i], transfer_sizes[i]});
        }
    }

public:
    std::string type() const noexcept {
        return type_;
    }

    gpu_id_t num_gpus() const noexcept {
        return num_gpus_;
    }

    gpu_id_t main_gpu() const noexcept {
        return main_gpu_;
    }

    void main_gpu(const gpu_id_t gpu) {
        main_gpu_ = gpu;
    }

    size_t num_steps() const noexcept {
        return num_steps_;
    }

    size_t num_chunks() const noexcept {
        return num_chunks_;
    }

    const std::vector<transfer_sequence>& transfer_sequences() const {
        return transfer_sequences_;
    }

    const std::vector<size_t>& sync_steps() {
        return sync_steps_;
    }

    void sync_steps(const std::vector<size_t>& steps) {
        sync_steps_ = steps;
    }

    bool synchronized() const noexcept {
        return sync_steps_.size() > 0;
    }

    bool valid() const noexcept {
        return valid_;
    }

    void validate() {
        valid_ = true;
    }

    void invalidate() {
        valid_ = false;
    }

    void show_plan() const {
        if(!valid_)
            std::cout << "ERROR: invalid plan\n";

        std::cout << "INFO: Transfer plan for " << uint32_t(num_gpus_) << " gpus\n";
        std::cout << "INFO: Transfer " << uint32_t(num_chunks_) << " chunks in " << num_steps_ << " steps\n";

        if(synchronized()) {
            std::cout << "INFO: Plan synchronizes after steps ";
            for(const auto& s : sync_steps_)
                std::cout << s << ' ';
            std::cout << '\n';
        }
        else {
            std::cout << "INFO: Plan is without synchronization\n";
        }

        for (const auto& sequence : transfer_sequences_) {
            std::cout << "\tTransfer "
                      << sequence.size
                      << " chunks via [";
            for(const auto& item : sequence.seq)
                std::cout << uint32_t(item) << ' ';
            std::cout << "]\n";
        }
        std::cout << std::endl;
    }

};

} // namespace

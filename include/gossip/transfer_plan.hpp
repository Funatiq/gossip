#pragma once

#include <vector>
#include <iostream>

#include "config.h"

namespace gossip {

template<
    bool throw_exceptions=true>
class transfer_plan_t {

protected:
    gpu_id_t num_gpus;
    size_t num_steps;
    std::vector<std::vector<gpu_id_t>> transfer_sequences;

    size_t num_chunks;
    std::vector<size_t> transfer_sizes;

    bool synchronized;
    bool valid;

public:
    transfer_plan_t(const gpu_id_t num_gpus_,
                    const std::vector<std::vector<gpu_id_t>>& transfer_sequences_ = {})
                    : num_gpus(num_gpus_),
                      num_steps(0),
                      transfer_sequences(transfer_sequences_),
                      num_chunks(1),
                      transfer_sizes(num_gpus, num_chunks),
                      synchronized(false),
                      valid(false)
    {}

    transfer_plan_t(const gpu_id_t num_gpus_,
                    const std::vector<std::vector<gpu_id_t>>& transfer_sequences_,
                    const size_t num_chunks_,
                    const std::vector<size_t>& transfer_sizes_)
                    : num_gpus(num_gpus_),
                      num_steps(0),
                      transfer_sequences(transfer_sequences_),
                      num_chunks(num_chunks_),
                      transfer_sizes(transfer_sizes_),
                      synchronized(false),
                      valid(false)
    {}

private:
    virtual void load_default_plan() = 0;

    virtual bool verify_plan() const = 0;

public:
    bool is_valid() const noexcept {
        return valid;
    }

    bool is_synchronized() const noexcept {
        return synchronized;
    }

    gpu_id_t get_num_gpus() const noexcept {
        return num_gpus;
    }

    size_t get_num_steps() const noexcept {
        return num_steps;
    }

    size_t get_num_chunks() const noexcept {
        return num_chunks;
    }

    const std::vector<std::vector<gpu_id_t>>& get_transfer_sequences() const {
        return transfer_sequences;
    }

    const std::vector<size_t>& get_transfer_sizes() const {
        return transfer_sizes;
    };

    void show_plan() const {
        if(!valid)
            std::cout << "WARNING: invalid plan\n";

        std::cout << "Transfer plan for " << int(num_gpus) << " gpus\n";
        std::cout << "Transfer " << num_chunks << " chunks";
        if(synchronized)
            std::cout << " in " << num_steps << " synchronized steps:\n";
        else
            std::cout << " asynchronously:\n";

        for (size_t i = 0; i < transfer_sequences.size(); ++i) {
            std::cout << "\tTransfer "
                      << ((num_chunks <= 1) ? 1 : transfer_sizes[i])
                      << " chunks via [";
            for(const auto& item : transfer_sequences[i])
                std::cout << int(item) << ' ';
            std::cout << "]\n";
        }
        std::cout << std::endl;
    }

};

} // namespace

#pragma once

#include <vector>
#include <iostream>

template<
    typename gpu_id_t,
    bool throw_exceptions=true>
class transfer_plan_t {

    gpu_id_t num_gpus;
    size_t num_steps;
    size_t num_chunks;

    std::vector<std::vector<gpu_id_t>> transfer_sequences;
    std::vector<size_t> transfer_sizes;

    bool valid;

public:
    transfer_plan_t(const gpu_id_t num_gpus_,
                    const std::vector<std::vector<gpu_id_t>>& transfer_sequences_ = {})
                    : num_gpus(num_gpus_),
                      num_chunks(1),
                      transfer_sequences(transfer_sequences_),
                      transfer_sizes(),
                      valid(false)
    {
        if(transfer_sequences.empty()) load_default_plan();
        valid = verify_plan();
    }

    transfer_plan_t(const gpu_id_t num_gpus_,
                    const std::vector<std::vector<gpu_id_t>>& transfer_sequences_,
                    const size_t num_chunks_,
                    const std::vector<size_t>& transfer_sizes_)
                    : num_gpus(num_gpus_),
                      num_chunks(num_chunks_),
                      transfer_sequences(transfer_sequences_),
                      transfer_sizes(transfer_sizes_),
                      valid(false)
    {
        if(transfer_sequences.empty()) load_default_plan();
        valid = verify_plan();
    }

private:
    void load_default_plan() {
        num_steps = 1;
        num_chunks = 1;

        transfer_sequences.reserve(num_gpus*num_gpus);

        // plan direct transfers from src to trg gpu
        for (gpu_id_t src = 0; src < num_gpus; ++src) {
            for (gpu_id_t trg = 0; trg < num_gpus; ++trg) {
                transfer_sequences.emplace_back(std::vector<gpu_id_t>{src,trg});
            }
        }
    }

    bool verify_plan() {
        num_steps = transfer_sequences[0].size()-1;

        if (num_steps < 1)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "planned sequence must be at least of length 2.");
            else return false;

        for (const auto& sequence : transfer_sequences)
            if (sequence.size() != num_steps+1)
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "planned sequences must have same lengths.");
                else return false;

        std::vector<std::vector<size_t> > completeness(num_gpus, std::vector<size_t>(num_gpus));
        if (num_chunks <= 1) {
            for (const auto& sequence : transfer_sequences) {
                completeness[sequence.front()][sequence.back()] += 1;
            }
        }
        else {
            if (transfer_sequences.size() != transfer_sizes.size())
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "number of sequences must match number of sizes.");
                else return false;
           for (size_t i = 0; i < transfer_sequences.size(); ++i) {
                completeness[transfer_sequences[i].front()][transfer_sequences[i].back()]
                    += transfer_sizes[i];
            }           
        }
        for (gpu_id_t src = 0; src < num_gpus; ++src) {
            for (gpu_id_t trg = 0; trg < num_gpus; ++trg) {
                if (completeness[src][trg] != num_chunks)
                    if (throw_exceptions)
                        throw std::invalid_argument(
                            "transfer plan is incomplete.");
                    else return false;
            }
        }

        return true;
    }

public:
    bool is_valid() const noexcept {
        return valid;
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

        std::cout << "Transfer plan for " << num_gpus << " gpus\n";
        std::cout << "Transfer " << num_chunks << "chunks in " << num_steps << " steps:\n";

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
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

#include "plan_parser.hpp"
#include "json.hpp"
using json = nlohmann::json;

using gpu_id_t = gossip::gpu_id_t;

bool
parse_plan(const char* filename,
           gpu_id_t& num_gpus,
           gpu_id_t& main_gpu,
           size_t& num_steps,
           size_t& num_chunks,
           std::vector<std::vector<gpu_id_t>>& transfer_sequences,
           std::vector<size_t>& transfer_sizes) {
    std::ifstream ifs(filename);
    json json_plan;
    if(ifs.good())
        ifs >> json_plan;
    else {
        std::cerr << "error reading " << filename << std::endl;
        return false;
    }

    // get plan from json
    auto it = json_plan.find("num_gpus");
    if(it != json_plan.end())
        num_gpus = *it;

    it = json_plan.find("main_gpu");
    if(it != json_plan.end())
        main_gpu = *it;

    it = json_plan.find("num_steps");
    if(it != json_plan.end())
        num_steps = *it;

    it = json_plan.find("num_chunks");
    if(it != json_plan.end())
        num_chunks = *it;

    it = json_plan.find("plan");
    if(it != json_plan.end())
        for(const auto& seq : *it) {
            transfer_sequences.push_back(seq);
            //TODO cut surplus items from seq
        }

    it = json_plan.find("counts");
    if(it != json_plan.end())
        for(const auto& seq : *it) {
            transfer_sizes.push_back(seq);
        }

    return true;
}

gossip::all2all_plan_t<>
parse_all2all_plan(const char* filename) {
    gpu_id_t num_gpus = 0;
    gpu_id_t main_gpu = 0;
    size_t num_steps = 0;
    size_t num_chunks = 0;
    std::vector<std::vector<gpu_id_t>> transfer_sequences = {};
    std::vector<size_t> transfer_sizes = {};

    parse_plan(filename, num_gpus, main_gpu, num_steps, num_chunks,
               transfer_sequences, transfer_sizes);

    if(num_chunks <= 1) {
        return gossip::all2all_plan_t<>{num_gpus, transfer_sequences};
    }
    else {
        return gossip::all2all_plan_t<>{num_gpus, transfer_sequences, num_chunks, transfer_sizes};
    }
}

gossip::scatter_plan_t<>
parse_scatter_plan(const char* filename) {
    gpu_id_t num_gpus = 0;
    gpu_id_t main_gpu = 0;
    size_t num_steps = 0;
    size_t num_chunks = 0;
    std::vector<std::vector<gpu_id_t>> transfer_sequences = {};
    std::vector<size_t> transfer_sizes = {};

    parse_plan(filename, num_gpus, main_gpu, num_steps, num_chunks,
               transfer_sequences, transfer_sizes);

    if(num_chunks <= 1) {
        return gossip::scatter_plan_t<>{main_gpu, num_gpus, transfer_sequences};
    }
    else {
        return gossip::scatter_plan_t<>{main_gpu, num_gpus, transfer_sequences, num_chunks, transfer_sizes};
    }
}

gossip::gather_plan_t<>
parse_gather_plan(const char* filename) {
    gpu_id_t num_gpus = 0;
    gpu_id_t main_gpu = 0;
    size_t num_steps = 0;
    size_t num_chunks = 0;
    std::vector<std::vector<gpu_id_t>> transfer_sequences = {};
    std::vector<size_t> transfer_sizes = {};

    parse_plan(filename, num_gpus, main_gpu, num_steps, num_chunks,
               transfer_sequences, transfer_sizes);

    if(num_chunks <= 1) {
        return gossip::gather_plan_t<>{main_gpu, num_gpus, transfer_sequences};
    }
    else {
        return gossip::gather_plan_t<>{main_gpu, num_gpus, transfer_sequences, num_chunks, transfer_sizes};
    }
}

#include <iostream>
#include <fstream>
#include <vector>

#include "gossip/transfer_plan.hpp"
#include "plan_parser.hpp"
#include "json.hpp"
using json = nlohmann::json;

transfer_plan_t<gpu_id_t>
parse_plan(const char* filename) {
    std::ifstream ifs(filename);
    json json_plan;
    if(ifs.good())
        ifs >> json_plan;
    else {
        std::cerr << "error reading " << filename << std::endl;
        return false;
    }

    // get plan from json
    gpu_id_t num_gpus = 0;
    auto it = json_plan.find("num_gpus");
    if(it != json_plan.end())
        num_gpus = *it;

    size_t num_steps = 0;
    it = json_plan.find("num_steps");
    if(it != json_plan.end())
        num_steps = *it;

    size_t num_chunks = 0;
    it = json_plan.find("num_chunks");
    if(it != json_plan.end())
        num_chunks = *it;

    std::vector<std::vector<gpu_id_t>> transfer_sequences = {};
    it = json_plan.find("plan");
    if(it != json_plan.end())
        for(const auto& seq : *it) {
            transfer_sequences.push_back(seq);
        }

    std::vector<size_t> transfer_sizes = {};
    it = json_plan.find("counts");
    if(it != json_plan.end())
        for(const auto& seq : *it) {
            transfer_sizes.push_back(seq);
        }

    if(num_chunks > 0)
        return transfer_plan_t<gpu_id_t>{num_gpus, transfer_sequences, num_chunks, transfer_sizes};
    else
        return transfer_plan_t<gpu_id_t>{num_gpus, transfer_sequences};
}


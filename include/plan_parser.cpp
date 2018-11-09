#include <iostream>
#include <fstream>
#include <vector>

#include "plan_parser.hpp"
#include "json.hpp"
using json = nlohmann::json;

bool parse_plan(const char* filename,
                gpu_id_t& num_gpus,
                std::vector<std::vector<gpu_id_t>>& transfer_plan) {
    std::ifstream ifs(filename);
    json json_plan;
    ifs >> json_plan;

    // get plan from json
    num_gpus = json_plan["num_gpus"];
    size_t num_steps = json_plan["num_steps"];

    for(const auto& seq : json_plan["plan"]) {
        transfer_plan.push_back(seq);
    }

    // check plan
    if (num_gpus < 2) {
        std::cerr << "plan loaded from " << filename << " has invalid number of gpus.";
        return false;   
    }
    if (num_steps < 1) {
        std::cerr << "plan loaded from " << filename << " has invalid number of phases.";
        return false;   
    }
    if(transfer_plan.size() != num_gpus*num_gpus) {
        std::cerr << "plan loaded from " << filename << " has invalid length.";
        return false;   
    }
    for (const auto& sequence : transfer_plan)
        if (sequence.size() != num_steps+1) {
            std::cerr << "plan loaded from " << filename << " has invalid sequence.";
            return false;   
        }
            
    std::vector<std::vector<bool> > completeness(num_gpus, std::vector<bool>(num_gpus));
    for (const auto& sequence : transfer_plan) {
        completeness[sequence.front()][sequence.back()] = true;
    }
    for (gpu_id_t src = 0; src < num_gpus; ++src) {
        for (gpu_id_t trg = 0; trg < num_gpus; ++trg) {
            if (!completeness[src][trg]) {
                std::cerr << "plan loaded from " << filename << " has is incomplete.";
                return false;   
            }
        }
    }

    return true;
}

void show_plan(std::vector<std::vector<gpu_id_t>>& transfer_plan) {
    std::cout << "Transfer plan:\n";
    for(const auto& sequence : transfer_plan) {
        for(const auto& item : sequence)
            std::cout << int(item) << ' ';
        std::cout << '\n';
    }
    std::cout << std::endl;
}


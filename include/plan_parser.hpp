#pragma once

#include <vector>

using gpu_id_t = uint8_t;

bool parse_plan(const char* filename,
                gpu_id_t& num_gpus,
                std::vector<std::vector<gpu_id_t>>& transfer_plan);

void show_plan(std::vector<std::vector<gpu_id_t>>& transfer_plan);
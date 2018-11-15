#pragma once

#include "gossip/transfer_plan.hpp"

using gpu_id_t = uint8_t;

transfer_plan_t<gpu_id_t> parse_plan(const char* filename);

#pragma once

#include "gossip/transfer_plan.hpp"

gossip::all2all_plan_t<> parse_plan(const char* filename);

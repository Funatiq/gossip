#pragma once

#include "gossip/all_to_all_plan.hpp"
#include "gossip/scatter_plan.hpp"
#include "gossip/gather_plan.hpp"

gossip::all2all_plan_t<> parse_all2all_plan(const char* filename);
gossip::scatter_plan_t<> parse_scatter_plan(const char* filename);
gossip::gather_plan_t<> parse_gather_plan(const char* filename);

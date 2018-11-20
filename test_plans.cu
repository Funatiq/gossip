#include <iostream>
#include <vector>

#include "include/gossip.cuh"

#include "include/plan_parser.hpp"

int main () {
    // auto transfer_plan = gossip::all2all_plan_t<>(2, {{0,0},{0,1},{1,0},{1,1}, {0,0},{0,1},{1,0},{1,1}},
                                                //   2, {1,1,1,1,1,1,1,1});
    auto all2all_plan = parse_all2all_plan("all2all_plan.json");
    all2all_plan.show_plan();

    auto scatter_plan = parse_scatter_plan("scatter_plan.json");
    scatter_plan.show_plan();

    auto gather_plan = parse_gather_plan("gather_plan.json");
    gather_plan.show_plan();
}

#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>

#include "include/gossip.cuh"
#include "distributed.cuh"

#include "include/plan_parser.hpp"

int main () {
    using data_t = uint64_t;

    double security_factor = 1.5;

    uint64_t batch_size = 1UL << 25;
    uint64_t batch_size_secure = batch_size * security_factor;

    // auto transfer_plan = gossip::all2all_plan_t<>(2, {{0,0},{0,1},{1,0},{1,1}, {0,0},{0,1},{1,0},{1,1}},
                                                //   2, {1,1,1,1,1,1,1,1});
    auto transfer_plan = parse_all2all_plan("plan.json");

    auto num_gpus = transfer_plan.get_num_gpus();

    if(transfer_plan.is_valid()) {
        // show_plan(transfer_plan);

        auto context = new gossip::context_t<>(num_gpus);
        auto all2all = new gossip::all2all_t<>(context, transfer_plan);
        auto multisplit = new gossip::multisplit_t<>(context);
        auto point2point = new gossip::point2point_t<>(context);

        run<data_t>(context, all2all, multisplit, point2point,
            batch_size, batch_size_secure);

        context->sync_hard();
        delete all2all;
        delete multisplit;
        delete point2point;
        delete context;
    }
}

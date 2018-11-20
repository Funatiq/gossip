#include <iostream>
#include <cstdint>

#include "include/gossip.cuh"
#include "distributed.cuh"

#include "include/plan_parser.hpp"

template<typename data_t>
void all2all(const size_t batch_size, const size_t batch_size_secure) {
    // auto transfer_plan = gossip::all2all_plan_t<>(2, {{0,0},{0,1},{1,0},{1,1}, {0,0},{0,1},{1,0},{1,1}},
                                                //   2, {1,1,1,1,1,1,1,1});
    auto transfer_plan = parse_all2all_plan("all2all_plan.json");

    auto num_gpus = transfer_plan.get_num_gpus();

    if(transfer_plan.is_valid()) {
        // transfer_plan.show_plan();

        auto context = new gossip::context_t<>(num_gpus);
        auto all2all = new gossip::all2all_t<>(context, transfer_plan);
        auto multisplit = new gossip::multisplit_t<>(context);
        auto point2point = new gossip::point2point_t<>(context);

        run_multisplit_all2all<data_t>(
            context, all2all, multisplit, point2point,
            batch_size, batch_size_secure);

        context->sync_hard();
        delete all2all;
        delete multisplit;
        delete point2point;
        delete context;
    }
}

template<typename data_t>
void scatter(const size_t batch_size, const size_t batch_size_secure) {
    auto transfer_plan = gossip::scatter_plan_t<>(0, 2, {{0,0},{0,1},{0,0},{0,1}},
                                                  2, {1,1,1,1});
    // auto transfer_plan = parse_scatter_plan("scatter_plan.json");

    auto num_gpus = transfer_plan.get_num_gpus();

    if(transfer_plan.is_valid()) {
        // transfer_plan.show_plan();

        auto context = new gossip::context_t<>(num_gpus);
        auto scatter = new gossip::scatter_t<>(context, transfer_plan);
        auto multisplit = new gossip::multisplit_t<>(context);
        auto point2point = new gossip::point2point_t<>(context);

        run_multisplit_scatter<data_t>(
            context, scatter, multisplit, point2point,
            batch_size, batch_size_secure);

        context->sync_hard();
        delete scatter;
        delete multisplit;
        delete point2point;
        delete context;
    }
}

int main () {
    using data_t = uint64_t;

    double security_factor = 1.5;

    size_t batch_size = 1UL << 25;
    size_t batch_size_secure = batch_size * security_factor;

    // all2all<data_t>(batch_size, batch_size_secure);

    scatter<data_t>(batch_size, batch_size_secure);

}

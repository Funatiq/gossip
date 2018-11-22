#include <iostream>
#include <cstdint>

#include "include/gossip.cuh"
#include "distributed.cuh"

#include "include/plan_parser.hpp"

template<typename data_t>
void all2all(const size_t batch_size, const size_t batch_size_secure) {
    // auto transfer_plan = gossip::all2all_plan_t<>(2, {{0,0},{0,1},{1,0},{1,1}, {0,0},{0,1},{1,0},{1,1}},
                                                //   2, {1,1,1,1,1,1,1,1});
    auto transfer_plan = gossip::all2all_plan_t<>(2);
    // auto transfer_plan = parse_all2all_plan("all2all_plan.json");

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
void scatter_gather(const size_t batch_size, const size_t batch_size_secure) {
    auto scatter_plan = gossip::scatter_plan_t<>(0, 2, {{0,0},{0,1},{0,0},{0,1}},
                                                  2, {1,1,1,1});
    // auto scatter_plan = gossip::scatter_plan_t<>(0, 2);
    // auto scatter_plan = parse_scatter_plan("scatter_plan.json");

    auto gather_plan = gossip::gather_plan_t<>(0, 2, {{0,0},{1,0},{0,0},{1,0}},
        2, {1,1,1,1});
    // auto gather_plan = gossip::gather_plan_t<>(0, 2);
    // auto gather_plan = parse_gather_plan("gather_plan.json");

    auto num_gpus = scatter_plan.get_num_gpus();
    if(num_gpus != gather_plan.get_num_gpus()) {
        std::cout << "scatter and gather do not match" << std::endl;
        return;
    }

    if(scatter_plan.is_valid() && gather_plan.is_valid()) {
        // scatter_plan.show_plan();
        // gather_plan.show_plan();

        auto context = new gossip::context_t<>(num_gpus);
        auto point2point = new gossip::point2point_t<>(context);
        auto multisplit = new gossip::multisplit_t<>(context);
        auto scatter = new gossip::scatter_t<>(context, scatter_plan);
        auto gather = new gossip::gather_t<>(context, gather_plan);

        run_multisplit_scatter_gather<data_t>(
            context, point2point, multisplit, scatter, gather,
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

    size_t batch_size = 1UL << 5;
    size_t batch_size_secure = batch_size * security_factor;

    // all2all<data_t>(batch_size, batch_size_secure);

    scatter_gather<data_t>(batch_size, batch_size_secure);

}

#include <iostream>
#include <cstdint>
#include <memory>

#include "include/gossip.cuh"
#include "distributed.cuh"

#include "include/plan_parser.hpp"

// global context
//auto context = std::make_unique< gossip::context_t >(4);

template<typename data_t>
void all2all(const size_t batch_size, const size_t batch_size_secure) {

    auto transfer_plan = parse_plan("all2all_plan.json");
    gossip::all2all::verify_plan(transfer_plan);

    auto num_gpus = transfer_plan.num_gpus();

    if(transfer_plan.valid()) {
        // transfer_plan.show_plan();

        auto context = std::make_unique< gossip::context_t >(num_gpus);
        auto all2all = std::make_unique< gossip::all2all_t >(*context, transfer_plan);
        auto multisplit = std::make_unique< gossip::multisplit_t >(*context);
        auto point2point = std::make_unique< gossip::point2point_t >(*context);

        run_multisplit_all2all<data_t>(
            *context, *all2all, *multisplit, *point2point,
            batch_size, batch_size_secure);

        context->sync_hard();
    }
}

template<typename data_t>
void all2all_async(const size_t batch_size, const size_t batch_size_secure) {

    auto transfer_plan = parse_plan("all2all_plan.json");
    gossip::all2all::verify_plan(transfer_plan);

    auto num_gpus = transfer_plan.num_gpus();

    if(transfer_plan.valid()) {
        // transfer_plan.show_plan();

        auto context = std::make_unique< gossip::context_t >(num_gpus);
        auto all2all = std::make_unique< gossip::all2all_async_t >(*context, transfer_plan);
        auto multisplit = std::make_unique< gossip::multisplit_t >(*context);
        auto point2point = std::make_unique< gossip::point2point_t >(*context);

        run_multisplit_all2all_async<data_t>(
            *context, *all2all, *multisplit, *point2point,
            batch_size, batch_size_secure);

        context->sync_hard();
    }
}

template<typename data_t>
void scatter_gather(const size_t batch_size, const size_t batch_size_secure) {

    auto scatter_plan = parse_plan("scatter_plan.json");
    gossip::scatter::verify_plan(scatter_plan);

    auto gather_plan = parse_plan("gather_plan.json");
    gossip::gather::verify_plan(gather_plan);

    auto num_gpus = scatter_plan.num_gpus();
    if(num_gpus != gather_plan.num_gpus()) {
        std::cout << "scatter and gather num_gpus does not match" << std::endl;
        return;
    }

    auto main_gpu = scatter_plan.main_gpu();
    if(main_gpu != gather_plan.main_gpu()) {
        std::cout << "scatter and gather main_gpu does not match" << std::endl;
        return;
    }

    if(scatter_plan.valid() && gather_plan.valid()) {
        // scatter_plan.show_plan();
        // gather_plan.show_plan();

        auto context = std::make_unique< gossip::context_t >(num_gpus);
        auto point2point = std::make_unique< gossip::point2point_t >(*context);
        auto multisplit = std::make_unique< gossip::multisplit_t >(*context);
        auto scatter = std::make_unique< gossip::scatter_t >(*context, scatter_plan);
        auto gather = std::make_unique< gossip::gather_t >(*context, gather_plan);

        run_multisplit_scatter_gather<data_t>(
            *context, *point2point, *multisplit, *scatter, *gather,
            main_gpu,
            batch_size, batch_size_secure);

        context->sync_hard();
    }
}

template<typename data_t>
void broadcaster(const size_t batch_size, const size_t batch_size_secure) {

    auto transfer_plan = parse_plan("broadcast_plan.json");
    gossip::broadcast::verify_plan(transfer_plan);

    auto num_gpus = transfer_plan.num_gpus();

    if(transfer_plan.valid()) {
        // transfer_plan.show_plan();

        auto context = std::make_unique< gossip::context_t >(num_gpus);
        auto broadcast = std::make_unique< gossip::broadcast_t >(*context, transfer_plan);
        auto multisplit = std::make_unique< gossip::multisplit_t >(*context);
        auto point2point = std::make_unique< gossip::point2point_t >(*context);

        run_multisplit_broadcast<data_t>(
            *context, *point2point, *multisplit, *broadcast,
            batch_size, batch_size_secure);

        context->sync_hard();
    }
}

int main (int argc, char *argv[]) {
    using data_t = uint64_t;

    double security_factor = 1.5;

    size_t batch_size = 1UL << 5;
    if(argc == 2)
    {
        batch_size = std::atoll(argv[1]);
    }

    size_t batch_size_secure = batch_size * security_factor;

    std::cout << "RUN: all2all" << std::endl;
    all2all<data_t>(batch_size, batch_size_secure);

    std::cout << "RUN: all2all_async" << std::endl;
    all2all_async<data_t>(batch_size, batch_size_secure);

    std::cout << "RUN: scatter_gather" << std::endl;
    scatter_gather<data_t>(batch_size, batch_size_secure);

    // std::cout << "broadcast" << std::endl;
    // broadcaster<data_t>(batch_size, batch_size_secure);
}

#include <iostream>
#include <cstdint>
#include <memory>

#include "include/gossip.cuh"
#include "distributed.cuh"

#include "include/plan_parser.hpp"

template<typename data_t>
void all2all(const size_t batch_size, const size_t batch_size_secure) {

    auto transfer_plan = parse_all2all_plan("all2all_plan.json");

    auto num_gpus = transfer_plan.get_num_gpus();

    if(transfer_plan.is_valid()) {
        // transfer_plan.show_plan();

        auto context = std::make_unique< gossip::context_t<> >(num_gpus);
        auto all2all = std::make_unique< gossip::all2all_t<> >(context.get(), transfer_plan);
        auto multisplit = std::make_unique< gossip::multisplit_t<> >(context.get());
        auto point2point = std::make_unique< gossip::point2point_t<> >(context.get());

        run_multisplit_all2all<data_t>(
            context.get(), all2all.get(), multisplit.get(), point2point.get(),
            batch_size, batch_size_secure);

        context->sync_hard();
    }
}

template<typename data_t>
void all2all_async(const size_t batch_size, const size_t batch_size_secure) {

    auto transfer_plan = parse_all2all_plan("all2all_plan.json");

    auto num_gpus = transfer_plan.get_num_gpus();

    if(transfer_plan.is_valid()) {
        // transfer_plan.show_plan();

        auto context = std::make_unique< gossip::context_t<> >(num_gpus);
        auto all2all = std::make_unique< gossip::all2all_async_t<> >(context.get(), transfer_plan);
        auto multisplit = std::make_unique< gossip::multisplit_t<> >(context.get());
        auto point2point = std::make_unique< gossip::point2point_t<> >(context.get());

        run_multisplit_all2all<data_t>(
            context.get(), all2all.get(), multisplit.get(), point2point.get(),
            batch_size, batch_size_secure);

        context->sync_hard();
    }
}

template<typename data_t>
void scatter_gather(const size_t batch_size, const size_t batch_size_secure) {

    auto scatter_plan = parse_scatter_plan("scatter_plan.json");
    auto gather_plan = parse_gather_plan("gather_plan.json");

    auto num_gpus = scatter_plan.get_num_gpus();
    if(num_gpus != gather_plan.get_num_gpus()) {
        std::cout << "scatter and gather do not match" << std::endl;
        return;
    }

    if(scatter_plan.is_valid() && gather_plan.is_valid()) {
        // scatter_plan.show_plan();
        // gather_plan.show_plan();

        auto context = std::make_unique< gossip::context_t<> >(num_gpus);
        auto point2point = std::make_unique< gossip::point2point_t<> >(context.get());
        auto multisplit = std::make_unique< gossip::multisplit_t<> >(context.get());
        auto scatter = std::make_unique< gossip::scatter_t<> >(context.get(), scatter_plan);
        auto gather = std::make_unique< gossip::gather_t<> >(context.get(), gather_plan);

        run_multisplit_scatter_gather<data_t>(
            context.get(), point2point.get(), multisplit.get(), scatter.get(), gather.get(),
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

    all2all<data_t>(batch_size, batch_size_secure);

    all2all_async<data_t>(batch_size, batch_size_secure);

    scatter_gather<data_t>(batch_size, batch_size_secure);

}

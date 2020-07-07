#include <iostream>
#include <cstdint>
#include <memory>
#include <string>
#include <assert.h>

#include "include/gossip.cuh"
#include "executor.cuh"
#include "include/plan_parser.hpp"
#include "include/clipp/include/clipp.h"

template<typename data_t>
void all2all(
    gossip::transfer_plan_t transfer_plan,
    const size_t batch_size,
    const size_t batch_size_secure) {

    gossip::all2all::verify_plan(transfer_plan);

    auto num_gpus = transfer_plan.num_gpus();
    std::vector<gpu_id_t> device_ids(num_gpus, 0);

    if(transfer_plan.valid()) {

        auto context = gossip::context_t(device_ids);
        // context.print_connectivity_matrix();
        auto all2all = gossip::all2all_t(context, transfer_plan);
        auto multisplit = gossip::multisplit_t(context);
        auto point2point = gossip::point2point_t(context);

        run_multisplit_all2all<data_t>(
            context, all2all, multisplit, point2point,
            batch_size, batch_size_secure);

        context.sync_hard();
    }
}

template<typename data_t>
void all2all_async(
    gossip::transfer_plan_t transfer_plan,
    const size_t batch_size,
    const size_t batch_size_secure) {

    gossip::all2all::verify_plan(transfer_plan);

    auto num_gpus = transfer_plan.num_gpus();
    std::vector<gpu_id_t> device_ids(num_gpus, 0);

    if(transfer_plan.valid()) {

        auto context = gossip::context_t(device_ids);
        // context.print_connectivity_matrix();
        auto all2all = gossip::all2all_async_t(context, transfer_plan);
        auto multisplit = gossip::multisplit_t(context);
        auto point2point = gossip::point2point_t(context);

        run_multisplit_all2all_async<data_t>(
            context, all2all, multisplit, point2point,
            batch_size, batch_size_secure);

        context.sync_hard();
    }
}

template<typename data_t>
void scatter_gather(
    gossip::transfer_plan_t scatter_plan,
    gossip::transfer_plan_t gather_plan,
    const size_t batch_size,
    const size_t batch_size_secure) {

    gossip::scatter::verify_plan(scatter_plan);
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

    std::vector<gpu_id_t> device_ids(num_gpus, 0);

    if(scatter_plan.valid() && gather_plan.valid()) {

        auto context = gossip::context_t(device_ids);
        // context.print_connectivity_matrix();
        auto point2point = gossip::point2point_t(context);
        auto multisplit = gossip::multisplit_t(context);
        auto scatter = gossip::scatter_t(context, scatter_plan);
        auto gather = gossip::gather_t(context, gather_plan);

        run_multisplit_scatter_gather<data_t>(
            context, point2point, multisplit, scatter, gather,
            main_gpu,
            batch_size, batch_size_secure);

        context.sync_hard();
    }
}

template<typename data_t>
void broadcaster(
    gossip::transfer_plan_t transfer_plan,
    const size_t batch_size,
    const size_t batch_size_secure) {

    gossip::broadcast::verify_plan(transfer_plan);

    auto num_gpus = transfer_plan.num_gpus();
    std::vector<gpu_id_t> device_ids(num_gpus, 0);

    if(transfer_plan.valid()) {

        auto context = gossip::context_t(device_ids);
        // context.print_connectivity_matrix();
        auto broadcast = gossip::broadcast_t(context, transfer_plan);
        auto multisplit = gossip::multisplit_t(context);
        auto point2point = gossip::point2point_t(context);

        run_multisplit_broadcast<data_t>(
            context, point2point, multisplit, broadcast,
            batch_size, batch_size_secure);

        context.sync_hard();
    }
}

int main (int argc, char *argv[]) {
    using data_t = uint64_t; // base data type

    // parse args using https://github.com/muellan/clipp
    using namespace clipp;
    enum class mode {all2all, all2all_async, scatter_gather, broadcast, help};

    mode selected;
    double security_factor = 1.5;
    size_t data_size = 28;
    std::string plan_file, scatter_plan_file, gather_plan_file;

    auto cli =
    (
        (
            (
                (
                    (
                        command("all2all").set(selected, mode::all2all) |
                        command("all2all_async").set(selected, mode::all2all_async) |
                        command("broadcast").set(selected, mode::broadcast)
                    ),
                    value("transfer plan", plan_file)
                ) |
                (
                    command("scatter_gather").set(selected, mode::scatter_gather),
                    value("scatter plan", scatter_plan_file), value("gather plan", gather_plan_file)
                )
            ),
            option("--size", "-s") & value("size", data_size) % "data size (bytes log2) [default: 28]",
            option("--memory-factor") & value("factor", security_factor) % "memory security factor [default: 1.5]"
        ) |
        command("help").set(selected, mode::help)
    );

    if(parse(argc, argv, cli))
    {
        assert(data_size >= 4);
        data_size = 1UL << data_size;
        size_t data_size_secure = data_size * security_factor;

        // execute selected collective
        switch(selected)
        {
            case mode::all2all:
                std::cout << "RUN: all2all" << std::endl;
                all2all<data_t>(parse_plan(plan_file.c_str()), data_size, data_size_secure);
                break;
            case mode::all2all_async:
                std::cout << "RUN: all2all_async" << std::endl;
                all2all_async<data_t>(parse_plan(plan_file.c_str()), data_size, data_size_secure);
                break;
            case mode::broadcast:
                std::cout << "RUN: broadcast" << std::endl;
                broadcaster<data_t>(parse_plan(plan_file.c_str()), data_size, data_size_secure);
                break;
            case mode::scatter_gather:
                std::cout << "RUN: scatter_gather" << std::endl;
                scatter_gather<data_t>(parse_plan(scatter_plan_file.c_str()), parse_plan(gather_plan_file.c_str()), data_size, data_size_secure);
                break;
            case mode::help:
                std::cout << make_man_page(cli, "execute").
                prepend_section("DESCRIPTION", "    test gossip on uniformly distributed data");
                break;
        }
    }
    else
    {
         std::cout << usage_lines(cli, "execute") << '\n';
    }

}

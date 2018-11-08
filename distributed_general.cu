#include <iostream>
#include <cstdint>
#include <vector>
#include "include/gossip.cuh"
#include "distributed.cuh"

int main () {
    using data_t = uint64_t;
    using gpu_id_t = gossip::gpu_id_t;

    double security_factor = 1.5;

    uint64_t batch_size = 1UL << 25;
    uint64_t batch_size_secure = batch_size * security_factor;
    std::vector<gpu_id_t> device_ids = {0, 1};

    std::vector<std::vector<gpu_id_t>> transfer_plan = {{0,0},{0,1},{1,0},{1,1}};

    auto context = new gossip::context_t<>(device_ids);
    auto all2all = new gossip::all2all_t<>(context, transfer_plan);
    auto multisplit = new gossip::multisplit_t<>(context);
    auto point2point = new gossip::point2point_t<>(context);

    context->print_connectivity_matrix();

    run<data_t>(context, all2all, multisplit, point2point,
                batch_size, batch_size_secure);

    context->sync_hard();
    delete all2all;
    delete multisplit;
    delete point2point;
    delete context;
}

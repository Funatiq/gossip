#include <iostream>
#include <cstdint>
#include "include/gossip.cuh"
#include "distributed.cuh"

int main () {
    using data_t = uint64_t;
    using gpu_id_t = gossip::gpu_id_t;

    double security_factor = 1.5;

    uint64_t batch_size = 1UL << 29;
    uint64_t batch_size_secure = batch_size * security_factor;
    std::vector<gpu_id_t> device_ids = {0, 1, 2, 3, 4, 5, 6, 7};

    auto context = new gossip::context_t<>(device_ids);
    auto all2all = new gossip::all2all_t<>(context);
    auto multisplit = new gossip::multisplit_t<>(context);
    auto point2point = new gossip::point2point_t<>(context);

    std::vector<std::vector<gpu_id_t>> transfer_plan = {
        {0,0,0},{0,1,1},{0,2,2},{0,3,3},{0,4,4},{0,4,5},{0,4,6},{0,4,7},
        {1,0,0},{1,1,1},{1,2,2},{1,3,3},{1,5,4},{1,5,5},{1,5,6},{1,5,7},
        {2,0,0},{2,2,1},{2,2,2},{2,3,3},{2,6,4},{2,1,5},{2,6,6},{2,3,7},
        {3,3,0},{3,1,1},{3,2,2},{3,3,3},{3,0,4},{3,7,5},{3,2,6},{3,7,7},
        {4,0,0},{4,0,1},{4,0,2},{4,0,3},{4,4,4},{4,5,5},{4,6,6},{4,7,7},
        {5,1,0},{5,1,1},{5,1,2},{5,1,3},{5,4,4},{5,5,5},{5,6,6},{5,7,7},
        {6,2,0},{6,5,1},{6,2,2},{6,7,3},{6,4,4},{6,6,5},{6,6,6},{6,7,7},
        {7,4,0},{7,3,1},{7,6,2},{7,3,3},{7,7,4},{7,5,5},{7,6,6},{7,7,7}
    };

    run<data_t>(context, all2all, multisplit, point2point,
                batch_size, batch_size_secure, transfer_plan);

    context->sync_hard();
    delete all2all;
    delete multisplit;
    delete point2point;
    delete context;
}

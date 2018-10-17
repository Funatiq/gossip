#include <iostream>
#include <cstdint>
#include "include/gossip.cuh"
#include "distributed.cuh"

int main () {
    using data_t = uint64_t;
    using gpu_id_t = gossip::gpu_id_t;

    const gpu_id_t num_gpus = 8;
    double security_factor = 1.5;

    uint64_t batch_size = 1UL << 29;
    uint64_t batch_size_secure = batch_size * security_factor;
    std::vector<gpu_id_t> device_ids = {0, 1, 2, 3, 4, 5, 6, 7};

    auto context = new gossip::context_t<num_gpus>(device_ids);
    auto all2all = new gossip::all2all_dgx1v_t<num_gpus>(context);
    auto multisplit = new gossip::multisplit_t<num_gpus>(context);
    auto point2point = new gossip::point2point_t<num_gpus>(context);

    run<num_gpus, data_t>(context, all2all, multisplit, point2point,
                          batch_size, batch_size_secure);

    context->sync_hard();
    delete all2all;
    delete multisplit;
    delete point2point;
    delete context;
}

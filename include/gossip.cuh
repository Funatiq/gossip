# pragma once

#include <cstdint>
#include <random>
#include <cstring>
#include <stdexcept>
#include "cudahelpers/cuda_helpers.cuh"

namespace gossip {
    using gpu_id_t = uint8_t;

    #include "gossip/context.cuh"
    #include "gossip/auxiliary.cuh"
    #include "gossip/all_to_all.cuh"
    #include "gossip/all_to_all_dgx1v.cuh"
    #include "gossip/multisplit.cuh"
    #include "gossip/point_to_point.cuh"
    #include "gossip/memory_manager.cuh"

    // only for unit tests
    #include "gossip/experiment.cuh"
}

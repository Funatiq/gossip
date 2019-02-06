# pragma once

#include <cstdint>

#define THROW_EXCEPTIONS 1

namespace gossip {

    using gpu_id_t = uint16_t;
    // type of multisplit counters
    using cnter_t = uint64_t;

    enum class PEER_STATUS : uint8_t {
        SLOW = 0,
        DIAG = 1,
        FAST = 2
    };

} // namespace

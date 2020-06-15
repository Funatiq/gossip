#pragma once

#include <iostream>
#include <vector>
#include <stdexcept>

#include "config.h"

namespace gossip {

    inline
    bool check(bool statement, const char* message) {
        if(!statement) {
    #ifdef THROW_EXCEPTIONS
                throw std::invalid_argument(message);
    #else
                std::cerr << message << std::endl;
                return false;
    #endif
        }
        return true;
    }

    template<typename index_t>
    bool check_size(
        const size_t transfer_size,
        const index_t buffer_length
    ) {
        return check(transfer_size <= buffer_length,
                    "buffer not large enough for transfers.");
    }

    template<typename index_t>
    bool check_size(
        const std::vector<size_t >& transfer_sizes,
        const std::vector<index_t>& buffer_lengths
    ) {
        for (gpu_id_t i = 0; i < transfer_sizes.size(); ++i) {
            if (!check_size(transfer_sizes[i], buffer_lengths[i]))
                return false;
        }
        return true;
    }

} // namespace

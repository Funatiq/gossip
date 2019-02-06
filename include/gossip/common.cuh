#pragma once

#include <iostream>
#include <stdexcept>

#include "config.h"

namespace gossip {

    struct transfer {
        const gpu_id_t src_gpu;
        const size_t src_pos;
        const gpu_id_t trg_gpu;
        const size_t trg_pos;
        const size_t len;
        const cudaEvent_t* event_before;
        const cudaEvent_t* event_after;

        transfer(const gpu_id_t src_gpu,
                 const size_t src_pos,
                 const gpu_id_t trg_gpu,
                 const size_t trg_pos,
                 const size_t len,
                 const cudaEvent_t* event_before,
                 const cudaEvent_t* event_after) :
            src_gpu(src_gpu),
            src_pos(src_pos),
            trg_gpu(trg_gpu),
            trg_pos(trg_pos),
            len(len),
            event_before(event_before),
            event_after(event_after)
        {}

        void show() const {
            std::cout <<   "src:" << int(src_gpu)
                      << ", pos:" << src_pos
                      << ", trg:" << int(trg_gpu)
                      << ", pos:" << trg_pos
                      << ", len:" << len
                      << ", event before:" << (event_before ? event_before : 0)
                      << ", event after:" << (event_after ? event_after : 0)
                      << std::endl;
        }
    };

    void show_transfers(const std::vector<transfer>& transfers) {
        for(const transfer& t : transfers) {
            t.show();
        }
    }

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

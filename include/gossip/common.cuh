#pragma once

#include <iostream>

#include "config.h"
#include "context.cuh"

namespace gossip {
    // shared between scatter, gather, all_to_all_async

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
                 const cudaEvent_t* event_before = nullptr,
                 const cudaEvent_t* event_after = nullptr) :
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

    template<typename table_t>
    struct transfer_handler {
        const context_t * context;

        std::vector<std::vector<size_t> > src_offsets;
        std::vector<std::vector<size_t> > trg_offsets;
        std::vector<size_t> aux_offsets;

        const std::vector<std::vector<size_t> >& src_displacements;
        const std::vector<std::vector<table_t> >& sizes;

        size_t num_phases;
        std::vector<std::vector<transfer> > phases;

        size_t num_chunks;

        std::vector<cudaEvent_t*> events;

        transfer_handler(
            const context_t * context_,
            const std::vector<std::vector<size_t>>& src_displacements,
            const std::vector<std::vector<size_t>>& trg_displacements,
            const std::vector<std::vector<table_t>>& sizes,
            const size_t num_phases_,
            const size_t num_chunks_ = 1
        ) :
            context(context_),
            src_offsets(src_displacements),     // src offsets begin at src displacements
            trg_offsets(trg_displacements),     // trg offsets begin at trg displacements
            aux_offsets(context->get_num_devices()),
            src_displacements(src_displacements),
            sizes(sizes),
            num_phases(num_phases_),
            phases(num_phases),
            num_chunks(num_chunks_)
        {}

        transfer_handler(const transfer_handler&) = delete;
        transfer_handler(transfer_handler&&) = default;

        ~transfer_handler() {
            for(auto& e : events)
                cudaEventDestroy(*e);
        }

        bool push_back(
            const std::vector<gpu_id_t>& sequence,
            const size_t chunks = 1,
            const bool verbose = false
        ) {
            if(!check(sequence.size() == num_phases+1,
                      "sequence size does not match number of phases."))
                return false;

            size_t* src_offset = &src_offsets[sequence.front()][sequence.back()];
            const size_t size_per_chunk = SDIV(sizes[sequence.front()][sequence.back()], num_chunks);
            size_t transfer_size = size_per_chunk * chunks;
            // check bounds
            const size_t limit = src_displacements[sequence.front()][sequence.back()]
                               + sizes[sequence.front()][sequence.back()];
            if (*src_offset + transfer_size > limit)
                transfer_size = limit - *src_offset;

            size_t* trg_offset = nullptr;

            cudaEvent_t* event_before = nullptr;
            cudaEvent_t* event_after = nullptr;

            if (verbose)
                std::cout << "transfer from " << int(sequence.front())
                          << " to " << int(sequence.back())
                          << std::endl;

            if (sequence.front() == sequence.back()) { // src == trg
                // direct transfer (copy) in first phase
                const size_t phase = 0;
                const gpu_id_t src = sequence.front();
                const gpu_id_t trg = sequence.back();

                trg_offset = &trg_offsets[sequence.front()][sequence.back()];

                phases[phase].emplace_back(src, *src_offset,
                                           trg, *trg_offset,
                                           transfer_size,
                                           event_before, event_after);
                if (verbose) phases[phase].back().show();

                // advance offsets
                *src_offset += transfer_size;
                *trg_offset += transfer_size;
            }
            else { // src != trg
                const gpu_id_t final_trg = sequence.back();

                for (size_t phase = 0; phase < num_phases; ++phase) {
                    const gpu_id_t src = sequence[phase];
                    const gpu_id_t trg = sequence[phase+1];
                    // schedule transfer only if device changes
                    if (src != trg) {
                        if (trg != final_trg) {
                            // transfer to auxiliary memory
                            trg_offset = &aux_offsets[trg];
                            // create event after transfer for synchronization
                            event_after = new cudaEvent_t();
                            const gpu_id_t id = context->get_device_id(src);
                            cudaSetDevice(id);
                            cudaEventCreate(event_after);
                            events.push_back(event_after);
                        }
                        else {
                            // transfer to final memory position
                            trg_offset = &trg_offsets[sequence.front()][sequence.back()];
                            // final transfer does not need follow up event
                            event_after = nullptr;
                        }

                        phases[phase].emplace_back(src, *src_offset,
                                                   trg, *trg_offset,
                                                   transfer_size,
                                                   event_before, event_after);
                        if (verbose) phases[phase].back().show();

                        // advance offset
                        *src_offset += transfer_size;
                        // old target is new source
                        src_offset = trg_offset;
                        event_before = event_after;

                        if (trg == final_trg)
                            break;
                    }
                }
                // advance last offset
                if (trg_offset)
                    *trg_offset += transfer_size;
            }

            return true;
        }

        void show_phase(const size_t phase) const {
            for(const transfer& t : phases[phase]) {
                t.show();
            }
        }

        template<typename value_t>
        bool execute_phase(
            const size_t phase,
            const std::vector<value_t *>& srcs,
            const std::vector<value_t *>& dsts,
            const std::vector<value_t *>& bufs
        ) const {
            for(const transfer& t : phases[phase]) {
                const gpu_id_t src = context->get_device_id(t.src_gpu);
                const gpu_id_t trg = context->get_device_id(t.trg_gpu);
                const auto stream  = context->get_streams(t.src_gpu)[t.trg_gpu];
                cudaSetDevice(src);
                const size_t size = t.len * sizeof(value_t);
                value_t * from = (t.event_before == nullptr) ?
                                srcs[t.src_gpu] + t.src_pos :
                                bufs[t.src_gpu] + t.src_pos;
                value_t * to   = (t.event_after == nullptr) ?
                                dsts[t.trg_gpu] + t.trg_pos :
                                bufs[t.trg_gpu] + t.trg_pos;

                if(t.event_before != nullptr) cudaStreamWaitEvent(stream, *(t.event_before), 0);
                cudaMemcpyPeerAsync(to, trg, from, src, size, stream);
                if(t.event_after != nullptr) cudaEventRecord(*(t.event_after), stream);
            } CUERR

            return true;
        }
    };

} // namespace

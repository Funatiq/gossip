#pragma once

#include <random>

#include "config.h"

namespace gossip {

template <
    bool throw_exceptions=true>
class experiment_t {

    gpu_id_t num_gpus;
    const context_t<> * context;
    bool external_context;

public:

    experiment_t (
        const gpu_id_t num_gpus_)
        : external_context (false) {

        context = new context_t<>(num_gpus_);
        num_gpus = context->get_num_devices();
    }

    experiment_t (
        const std::vector<gpu_id_t>& device_ids_)
        : external_context (false) {

        context = new context_t<>(device_ids_);
        num_gpus = context->get_num_devices();
    }

    experiment_t (
        context_t<> * context_) : context(context_),
                                          external_context (true) {
            if (throw_exceptions)
                if (!context->is_valid())
                    throw std::invalid_argument(
                        "You have to pass a valid context!"
                    );

        num_gpus = context->get_num_devices();
    }

    ~experiment_t () {
        if (!external_context)
            delete context;
    }

    template <
        typename value_t,
        typename index_t>
    bool create_random_data_host(
        const std::vector<value_t *>& srcs_host,
        const std::vector<index_t  >& srcs_lens) const {

        if (srcs_host.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "srcs_host size does not match number of gpus.");
            else return false;
        if (srcs_lens.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "srcs_lens size does not match number of gpus.");
            else return false;

        // initialize RNG
        std::random_device rd;
        std::mt19937 engine(rd());
        std::uniform_int_distribution<uint64_t> rho;

        // fill the source vector according to the partition table
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu)
            for (index_t index = 0; index < srcs_lens[gpu]; ++index)
                srcs_host[gpu][index] = rho(engine) % (num_gpus+1);


        return true;
    }

    template <
        typename value_t,
        typename index_t,
        typename table_t>
    bool validate_all2all_host(
        const std::vector<value_t *>& dsts_host,
        const std::vector<index_t  >& dsts_lens,
        const std::vector<std::vector<table_t> >& table
    ) const {

        if (dsts_host.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "dsts_host size does not match number of gpus.");
            else return false;
        if (dsts_lens.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "dsts_lens size does not match number of gpus.");
            else return false;
        if (table.size() != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "table size does not match number of gpus.");
            else return false;   
        for (const auto& t : table)
            if (t.size() != num_gpus)
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "table size does not match number of gpus.");
                else return false;    

        // compute prefix sums over the partition table
        // vertical scan
        std::vector<std::vector<table_t> >
            v_table(num_gpus, std::vector<table_t>(num_gpus));
        
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            for (gpu_id_t part = 0; part < num_gpus; ++part) {
                v_table[gpu+1][part] = table[gpu][part]+v_table[gpu][part];
            }
        }

        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            uint64_t accum = 0;
            for (index_t i = 0; i < dsts_lens[gpu]; ++i)
                accum += dsts_host[gpu][i] == gpu+1;
            if (accum != v_table[num_gpus][gpu]) {
                std::cout << "ERROR: dsts entries differ from expectation "
                          << "(expected: " << v_table[num_gpus][gpu]
                          << " seen: " << accum << " )"
                          << std::endl;
                return false;
            }
        }
        return true;
    }
};

} // namespace

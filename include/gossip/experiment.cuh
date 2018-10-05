#pragma once

template <
    uint64_t num_gpus,
    uint64_t throw_exceptions=true>
class experiment_t {

    const context_t<num_gpus> * context;
    bool external_context;

public:

    experiment_t (
        uint64_t * device_ids_=0) : external_context (false) {

        if (device_ids_)
            context = new context_t<num_gpus>(device_ids_);
        else
            context = new context_t<num_gpus>();
    }

    experiment_t (
        context_t<num_gpus> * context_) : context(context_),
                                          external_context (true) {
            if (throw_exceptions)
                if (!context->is_valid())
                    throw std::invalid_argument(
                        "You have to pass a valid context!"
                    );
    }

    ~experiment_t () {
        if (!external_context)
            delete context;
    }

    template <
        typename value_t,
        typename index_t>
    bool create_random_data_host(
        value_t *srcs_host[num_gpus],
        index_t  srcs_lens[num_gpus]) const {

        // initialize RNG
        std::random_device rd;
        std::mt19937 engine(rd());
        std::uniform_int_distribution<uint64_t> rho;

        // fill the source array according to the partition table
        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu)
            for (uint64_t index = 0; index < srcs_lens[gpu]; ++index)
                srcs_host[gpu][index] = rho(engine) % (num_gpus+1);


        return true;
    }

    template <
        typename value_t,
        typename index_t>
    bool validate_all2all_host(
        value_t *dsts_host[num_gpus],
        index_t  dsts_lens[num_gpus],
        index_t table[num_gpus][num_gpus]) const {

        // compute prefix sums over the partition table
        uint64_t v_table[num_gpus+1][num_gpus] = {0}; // vertical scan

        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
            for (uint64_t part = 0; part < num_gpus; ++part) {
                v_table[gpu+1][part] = table[gpu][part]+v_table[gpu][part];
            }
        }

        for (uint64_t gpu = 0; gpu < num_gpus; ++gpu) {
            uint64_t accum = 0;
            for (uint64_t i = 0; i < dsts_lens[gpu]; ++i)
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

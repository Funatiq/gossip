#pragma once

template <
    gpu_id_t num_gpus,
    bool throw_exceptions=true>
class experiment_t {

    const context_t<num_gpus> * context;
    bool external_context;

public:

    experiment_t (
        std::vector<gpu_id_t>& device_ids_ = std::vector<gpu_id_t>{})
        : external_context (false) {

        context = new context_t<num_gpus>(device_ids_);
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
        const std::array<std::array<value_t *, num_gpus>, num_gpus>& srcs_host,
        const std::array<index_t, num_gpus>& srcs_lens) const {

        // initialize RNG
        std::random_device rd;
        std::mt19937 engine(rd());
        std::uniform_int_distribution<uint64_t> rho;

        // fill the source array according to the partition table
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
        const std::array<std::array<value_t *, num_gpus>, num_gpus>& dsts_host,
        const std::array<index_t, num_gpus>& dsts_lens,
        const std::array<std::array<table_t, num_gpus>, num_gpus>& table
    ) const {

        // compute prefix sums over the partition table
        // vertical scan
        std::array<std::array<table_t, num_gpus>, num_gpus+1> v_table = {};
        
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

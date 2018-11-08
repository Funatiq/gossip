#pragma once

namespace dgx1v {
    const std::vector<std::vector<gpu_id_t> > default_plan = {
        {0,0,0},{0,1,1},{0,2,2},{0,3,3},{0,4,4},{0,4,5},{0,4,6},{0,4,7},
        {1,0,0},{1,1,1},{1,2,2},{1,3,3},{1,5,4},{1,5,5},{1,5,6},{1,5,7},
        {2,0,0},{2,2,1},{2,2,2},{2,3,3},{2,6,4},{2,1,5},{2,6,6},{2,3,7},
        {3,3,0},{3,1,1},{3,2,2},{3,3,3},{3,0,4},{3,7,5},{3,2,6},{3,7,7},
        {4,0,0},{4,0,1},{4,0,2},{4,0,3},{4,4,4},{4,5,5},{4,6,6},{4,7,7},
        {5,1,0},{5,1,1},{5,1,2},{5,1,3},{5,4,4},{5,5,5},{5,6,6},{5,7,7},
        {6,2,0},{6,5,1},{6,2,2},{6,7,3},{6,4,4},{6,6,5},{6,6,6},{6,7,7},
        {7,4,0},{7,3,1},{7,6,2},{7,3,3},{7,7,4},{7,5,5},{7,6,6},{7,7,7}
    };
}

template<
    bool throw_exceptions=true>
class all2all_dgx1v_t : public all2all_t<throw_exceptions> {

    static constexpr gpu_id_t num_gpus = 8;

public:
    all2all_dgx1v_t ()
        : all2all_t<throw_exceptions>(num_gpus, dgx1v::default_plan)
    {};

    all2all_dgx1v_t (
        const std::vector<std::vector<gpu_id_t> >& transfer_plan_)
        : all2all_t<throw_exceptions>(num_gpus, transfer_plan_)
    {};

    all2all_dgx1v_t (
        context_t<> * context_)
        : all2all_t<throw_exceptions>(context_, dgx1v::default_plan)
    {
        check_context();
    };

    all2all_dgx1v_t (
        context_t<> * context_,
        const std::vector<std::vector<gpu_id_t> >& transfer_plan_)
        : all2all_t<throw_exceptions>(context_, transfer_plan_)
    {
        check_context();
    };

private:
    bool check_context() const {
        if (this->num_gpus != num_gpus)
            if (throw_exceptions)
                throw std::invalid_argument(
                    "Context is invalid for DGX1V!"
                );
            else return false;
        for(gpu_id_t gpu = 0; gpu < this->num_gpus; ++gpu)
            if (this->context->get_device_id(gpu) != gpu)
                if (throw_exceptions)
                    throw std::invalid_argument(
                        "Context is invalid for DGX1V!"
                    );
            else return false;

        return true;
    }
};

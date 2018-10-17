#pragma once

template <
    gpu_id_t num_gpus,
    bool throw_exceptions=true,
    uint64_t PEER_STATUS_SLOW=0,
    uint64_t PEER_STATUS_DIAG=1,
    uint64_t PEER_STATUS_FAST=2>
class context_t {

    cudaStream_t * streams;
    gpu_id_t * device_ids;
    uint64_t peer_status[num_gpus][num_gpus];

public:

    context_t (gpu_id_t * device_ids_=0) {

        // copy num_gpus many device identifiers
        device_ids = new gpu_id_t[num_gpus];
        for (gpu_id_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
            device_ids[src_gpu] = device_ids_ ?
                                  device_ids_[src_gpu] : src_gpu;

        // create num_gpus^2 streams where streams[gpu*num_gpus+part]
        // denotes the stream to be used for GPU gpu and partition part
        streams = new cudaStream_t[num_gpus*num_gpus];
        for (gpu_id_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
            cudaSetDevice(get_device_id(src_gpu));
            cudaDeviceSynchronize();
            for (gpu_id_t part = 0; part < num_gpus; ++part) {
                cudaStreamCreate(get_streams(src_gpu)+part);
            }
        } CUERR


        // compute the connectivity matrix
        for (gpu_id_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
            const gpu_id_t src = get_device_id(src_gpu);
            cudaSetDevice(src);
            for (gpu_id_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                const gpu_id_t dst = get_device_id(dst_gpu);

                // check if src can access dst
                if (src == dst) {
                    peer_status[src_gpu][dst_gpu] = PEER_STATUS_DIAG;
                } else {
                    int32_t status;
                    cudaDeviceCanAccessPeer(&status, src, dst);
                    peer_status[src_gpu][dst_gpu] = status ?
                                                    PEER_STATUS_FAST :
                                                    PEER_STATUS_SLOW ;
                }
            }
        } CUERR

        for (gpu_id_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
            const gpu_id_t src = get_device_id(src_gpu);
            cudaSetDevice(src);
            for (gpu_id_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                const gpu_id_t dst = get_device_id(dst_gpu);

                if (src_gpu != dst_gpu) {
                    if (throw_exceptions)
                        if (src == dst)
                            throw std::invalid_argument(
                                "Device identifiers are not unique.");
                }

                if (peer_status[src_gpu][dst_gpu] == PEER_STATUS_FAST) {
                    cudaDeviceEnablePeerAccess(dst, 0);

                    // consume error for rendundant
                    // peer access initialization
                    const cudaError_t cuerr = cudaGetLastError();
                    if (cuerr == cudaErrorPeerAccessAlreadyEnabled)
                        std::cout << "STATUS: redundant enabling of "
                                  << "peer access from GPU " << src
                                  << " to GPU " << dst << " attempted."
                                  << std::endl;
                    else if (cuerr)
                        std::cout << "CUDA error: "
                                  << cudaGetErrorString(cuerr) << " : "
                                  << __FILE__ << ", line "
                                  << __LINE__ << std::endl;
                }

            }
        } CUERR
    }

    ~context_t () {

        // synchronize and destroy streams
        for (gpu_id_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
            cudaSetDevice(get_device_id(src_gpu));
            cudaDeviceSynchronize();
            for (gpu_id_t part = 0; part < num_gpus; ++part) {
                cudaStreamSynchronize(get_streams(src_gpu)[part]);
                cudaStreamDestroy(get_streams(src_gpu)[part]);
            }
        } CUERR

        // disable peer access
        for (gpu_id_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
            const gpu_id_t src = get_device_id(src_gpu);
            cudaSetDevice(src);
            for (gpu_id_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu) {
                const gpu_id_t dst = get_device_id(dst_gpu);

                if (peer_status[src_gpu][dst_gpu] == PEER_STATUS_FAST) {
                    cudaDeviceDisablePeerAccess(dst);

                    // consume error for rendundant
                    // peer access deactivation
                    const cudaError_t cuerr = cudaGetLastError();
                    if (cuerr == cudaErrorPeerAccessNotEnabled)
                        std::cout << "STATUS: redundant disabling of "
                                  << "peer access from GPU " << src_gpu
                                  << " to GPU " << dst << " attempted."
                                  << std::endl;
                    else if (cuerr)
                        std::cout << "CUDA error: "
                                  << cudaGetErrorString(cuerr) << " : "
                                   << __FILE__ << ", line "
                                   << __LINE__ << std::endl;
                }
            }
        } CUERR

        // free streams and device identifiers
        delete [] streams;
        delete [] device_ids;
    }

    gpu_id_t get_device_id (const gpu_id_t gpu) const noexcept {

        // return the actual device identifier of GPU gpu
        return device_ids[gpu];
    }

    cudaStream_t * get_streams (const gpu_id_t gpu) const noexcept {

        // return pointer to all num_gpus many streams of GPU gpu
        return streams+gpu*num_gpus;
    }

    void sync_gpu_streams (const gpu_id_t gpu) const noexcept {

        // sync all streams associated with the corresponding GPU
        cudaSetDevice(get_device_id(gpu)); CUERR
        for (gpu_id_t part = 0; part < num_gpus; ++part)
            cudaStreamSynchronize(get_streams(gpu)[part]);
        CUERR
    }

    void sync_all_streams () const noexcept {

        // sync all streams of the context
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu)
            sync_gpu_streams(gpu);
        CUERR
    }

    void sync_hard () const noexcept {

        // sync all GPUs
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(get_device_id(gpu));
            cudaDeviceSynchronize();
        } CUERR
    }

    bool is_valid () const noexcept {

        // both streams and device identifiers are created
        return streams && device_ids;
    }

    void print_connectivity_matrix () const {
        std::cout << "STATUS: connectivity matrix:" << std::endl;
        for (gpu_id_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu)
            for (gpu_id_t dst_gpu = 0; dst_gpu < num_gpus; ++dst_gpu)
                std::cout << (dst_gpu == 0 ? "STATUS: |" : "")
                          << uint64_t(peer_status[src_gpu][dst_gpu])
                          << (dst_gpu+1 == num_gpus ? "|\n" : " ");
    }
};

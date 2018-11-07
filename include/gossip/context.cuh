#pragma once

#include <numeric>

template <
    bool throw_exceptions=true,
    uint64_t PEER_STATUS_SLOW=0,
    uint64_t PEER_STATUS_DIAG=1,
    uint64_t PEER_STATUS_FAST=2>
class context_t {

    gpu_id_t num_gpus;
    std::vector<gpu_id_t> device_ids;
    std::vector<std::vector<cudaStream_t>> streams;
    std::vector<std::vector<uint64_t>> peer_status;
    bool valid = true;

public:

    context_t (const gpu_id_t num_gpus_) {

        if(num_gpus_ < 1) {
            valid = false;
            if (throw_exceptions)
                throw std::invalid_argument(
                    "Invalid number of devices.");
        }

        num_gpus = num_gpus_;

        device_ids.resize(num_gpus);
        std::iota(device_ids.begin(), device_ids.end(), 0);

        initialize();
    }

    context_t (std::vector<gpu_id_t>& device_ids_) {

        if(device_ids_.empty()) {
            valid = false;
            if (throw_exceptions)
                throw std::invalid_argument(
                    "Invalid number of device ids.");
        }

        num_gpus = device_ids_.size();  
        
        device_ids = device_ids_;

        initialize();
    }

private:
    void initialize() {
        if(!valid) return;

        streams.resize(num_gpus, std::vector<cudaStream_t>(num_gpus));
        
        // create num_gpus^2 streams where streams[gpu*num_gpus+part]
        // denotes the stream to be used for GPU gpu and partition part
        for (gpu_id_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
            cudaSetDevice(get_device_id(src_gpu));
            cudaDeviceSynchronize();
            for (gpu_id_t part = 0; part < num_gpus; ++part) {
                cudaStreamCreate(&streams[src_gpu][part]);
            }
        } CUERR

        peer_status.resize(num_gpus, std::vector<uint64_t>(num_gpus));

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
    
public:
    ~context_t () {

        if(!valid) return;

        // synchronize and destroy streams
        for (gpu_id_t src_gpu = 0; src_gpu < num_gpus; ++src_gpu) {
            cudaSetDevice(get_device_id(src_gpu));
            cudaDeviceSynchronize();
            for (gpu_id_t part = 0; part < num_gpus; ++part) {
                cudaStreamSynchronize(get_streams(src_gpu)[part]);
                cudaStreamDestroy(streams[src_gpu][part]);
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
    }

    // return the number of devices belonging to context
    gpu_id_t get_num_devices () const noexcept {
        return num_gpus;
    }

    // return the actual device identifier of specified GPU
    gpu_id_t get_device_id (const gpu_id_t gpu) const noexcept {
        return device_ids[gpu];
    }

    // return vector of streams associated with to specified GPU
    const std::vector<cudaStream_t>& get_streams (const gpu_id_t gpu) const noexcept {
        return streams[gpu];
    }

    // sync all streams associated with the specified GPU
    void sync_gpu_streams (const gpu_id_t gpu) const noexcept {
        cudaSetDevice(get_device_id(gpu)); CUERR
        for (gpu_id_t part = 0; part < num_gpus; ++part)
            cudaStreamSynchronize(get_streams(gpu)[part]);
        CUERR
    }

    // sync all streams of the context
    void sync_all_streams () const noexcept {
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu)
            sync_gpu_streams(gpu);
        CUERR
    }

    // sync all GPUs
    void sync_hard () const noexcept {
        for (gpu_id_t gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(get_device_id(gpu));
            cudaDeviceSynchronize();
        } CUERR
    }

    // check if both streams and device identifiers are created
    bool is_valid () const noexcept {
        return !streams.empty() && !device_ids.empty();
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

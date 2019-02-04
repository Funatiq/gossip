#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include "../include/cudahelpers/cuda_helpers.cuh"

template <typename It>
void print(It begin, It end)
{
    while(begin < end)
    {
        std::cout << *begin << std::endl;
        begin++;
    }
}

template <typename It>
void print_statistic(It begin, It end)
{
    using T = typename std::iterator_traits<It>::value_type;
    std::vector<T> data(begin, end);

    T min = T(*std::min_element(data.begin(), data.end()));
    T max = T(*std::max_element(data.begin(), data.end()));

    std::nth_element(data.begin(), data.begin() + data.size() / 2, data.end());

    T median = data[data.size() / 2];
    
    std::cout << "min=" << min << " max=" << max << " median=" << median << std::endl;
}

int main (int argc, char *argv[])
{
    if(argc != 3)
    {
        std::cerr << "invalid number of arguments" << std::endl;
        exit(1);
    }

    using data_t = char;
    using index_t = std::uint64_t;

    std::vector<index_t> sizes = {
        4194304,
        8388608,
        16777216,
        33554432,
        67108864,
        134217728,
        268435456,
        536870912,
        1073741824};

    index_t max_size = index_t(*std::max_element(sizes.begin(), sizes.end()));
    index_t repeats = 100;

    index_t gpu_x_id = atoi(argv[1]);
    index_t gpu_y_id = atoi(argv[2]);

    data_t* gpu_x_buffer_a = nullptr;
    data_t* gpu_x_buffer_b = nullptr;
    data_t* gpu_y_buffer_a = nullptr;
    data_t* gpu_y_buffer_b = nullptr;

    cudaEvent_t start, stop;
    cudaStream_t gpu_x_stream, gpu_y_stream;
    std::vector<float> timings(repeats);

    // init
    cudaSetDevice(gpu_x_id); CUERR
    cudaDeviceEnablePeerAccess(gpu_y_id, 0); CUERR
    cudaEventCreate(&start); CUERR
    cudaEventCreate(&stop); CUERR
    cudaStreamCreate(&gpu_x_stream); CUERR
    cudaMalloc(&gpu_x_buffer_a, sizeof(data_t)*max_size); CUERR
    cudaMalloc(&gpu_x_buffer_b, sizeof(data_t)*max_size); CUERR

    cudaSetDevice(gpu_y_id); CUERR
    cudaDeviceEnablePeerAccess(gpu_x_id, 0); CUERR
    cudaStreamCreate(&gpu_y_stream); CUERR
    cudaMalloc(&gpu_y_buffer_a, sizeof(data_t)*max_size); CUERR
    cudaMalloc(&gpu_y_buffer_b, sizeof(data_t)*max_size); CUERR

    
    for(auto size : sizes)
    {
        for(index_t i = 0; i < repeats; i++)
        {
            cudaSetDevice(gpu_x_id); CUERR
            cudaEventRecord(start);
            cudaMemcpyAsync(gpu_x_buffer_a, gpu_y_buffer_a, sizeof(data_t)*size, D2D, gpu_x_stream);
            cudaStreamSynchronize(gpu_x_stream);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&(timings[i]), start, stop); CUERR
        }
        std::cout << "unidirectional: size=" << size*sizeof(data_t) << std::endl;
        //print(timings.begin(), timings.end());
        print_statistic(timings.begin(), timings.end());

        for(index_t i = 0; i < repeats; i++)
        {
            cudaSetDevice(gpu_x_id); CUERR
            cudaEventRecord(start);
            cudaMemcpyAsync(gpu_x_buffer_a, gpu_y_buffer_a, sizeof(data_t)*size, D2D, gpu_x_stream);
            cudaSetDevice(gpu_y_id);
            cudaMemcpyAsync(gpu_y_buffer_b, gpu_x_buffer_b, sizeof(data_t)*size, D2D, gpu_y_stream);
            cudaStreamSynchronize(gpu_x_stream);
            cudaStreamSynchronize(gpu_y_stream);
            cudaSetDevice(gpu_x_id);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&(timings[i]), start, stop); CUERR
        }
        std::cout << "bidirectional size=2*" << size*sizeof(data_t) << std::endl;
        //print(timings.begin(), timings.end());
        print_statistic(timings.begin(), timings.end());
    }

    // cleanup
    cudaSetDevice(gpu_x_id); CUERR
    cudaEventDestroy(start); CUERR
    cudaEventDestroy(stop); CUERR
    cudaStreamDestroy(gpu_x_stream); CUERR
    cudaFree(gpu_x_buffer_a); CUERR
    cudaFree(gpu_x_buffer_b); CUERR

    cudaSetDevice(gpu_y_id); CUERR
    cudaStreamDestroy(gpu_y_stream); CUERR
    cudaFree(gpu_y_buffer_a); CUERR
    cudaFree(gpu_y_buffer_b); CUERR
}
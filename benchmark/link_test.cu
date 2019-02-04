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
void print_csv(It begin, It end)
{
    while(begin < end-1)
    {
        std::cout << *begin << ",";
        begin++;
    }
    std::cout << *begin << std::endl;
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
    /*
    std::vector<index_t> sizes = {
        10923,
        21846,
        43691,
        87382,
        174763,
        349526,
        699051,
        1398102,
        2796203,
        5592406,
        11184811,
        22369622,
        44739243,
        89478486,
        178956971,
        357913942,
        715827883,
        1431655766};
    */

    std::vector<index_t> sizes;
    for(index_t i = 11; i < 34; i++)
    {
        sizes.push_back(1ULL << i);
        //std::cout << sizes.back() << std::endl;
    }
    
    index_t max_size = index_t(*std::max_element(sizes.begin(), sizes.end()));
    index_t repeats = 10;

    index_t src_id = atoi(argv[1]);
    index_t dst_id = atoi(argv[2]);

    data_t* src_ptr = nullptr;
    data_t* dst_ptr = nullptr;

    cudaEvent_t start, stop;
    std::vector<float> timings(repeats);

    // init
    cudaSetDevice(src_id); CUERR
    cudaDeviceEnablePeerAccess(dst_id, 0); CUERR
    cudaEventCreate(&start); CUERR
    cudaEventCreate(&stop); CUERR
    cudaMalloc(&src_ptr, sizeof(data_t)*max_size); CUERR

    cudaSetDevice(dst_id); CUERR
    cudaDeviceEnablePeerAccess(src_id, 0); CUERR
    cudaMalloc(&dst_ptr, sizeof(data_t)*max_size); CUERR

    // timer resolution
    cudaSetDevice(src_id); CUERR
    for(index_t i = 0; i < repeats; i++)
    {
        cudaSetDevice(src_id); CUERR
        cudaEventRecord(start);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&(timings[i]), start, stop); CUERR
    }
    //std::cout << "timer resolution" << std::endl;
    //print_csv(timings.begin(), timings.end());
    //print_statistic(timings.begin(), timings.end());

    // call latency
    cudaSetDevice(src_id); CUERR
    for(index_t i = 0; i < repeats; i++)
    {
        cudaEventRecord(start);
        cudaMemcpy(src_ptr, dst_ptr, 0, D2D);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&(timings[i]), start, stop); CUERR
    }
    //std::cout << "call latencies" << std::endl;
    //print_csv(timings.begin(), timings.end());
    //print_statistic(timings.begin(), timings.end());

    // memcpy latency
    cudaSetDevice(src_id); CUERR
    for(index_t i = 0; i < repeats; i++)
    {
        cudaEventRecord(start);
        cudaMemcpy(src_ptr, dst_ptr, sizeof(data_t), D2D);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&(timings[i]), start, stop); CUERR
    }
    //std::cout << "memcpy latencies" << std::endl;
    //print_csv(timings.begin(), timings.end());
    //print_statistic(timings.begin(), timings.end());

    // memcpy bandwidth
    cudaSetDevice(src_id); CUERR
    for(auto size : sizes)
    {
        for(index_t i = 0; i < repeats; i++)
        {
            cudaEventRecord(start);
            cudaMemcpy(src_ptr, dst_ptr, sizeof(data_t)*size, D2D);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&(timings[i]), start, stop); CUERR
        }
        std::cout << size*sizeof(data_t) << ",";
        print_csv(timings.begin(), timings.end());
        //print_statistic(timings.begin(), timings.end());
    }

    // cleanup
    cudaSetDevice(src_id); CUERR
    cudaEventDestroy(start); CUERR
    cudaEventDestroy(stop); CUERR
    cudaFree(src_ptr); CUERR

    cudaSetDevice(dst_id); CUERR
    cudaFree(dst_ptr); CUERR
}
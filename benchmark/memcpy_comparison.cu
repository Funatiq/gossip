#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include "../include/cudahelpers/cuda_helpers.cuh"

template<class T>
GLOBALQUALIFIER
void memcpy_kernel(T* src, T* dst, std::uint64_t size)
{
    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid == 0)
    {
        memcpy(dst, src, size*sizeof(T));
    }
}

template<class T, std::uint64_t ChunkSize = 1>
GLOBALQUALIFIER
void memcpy2_kernel(T* src, T* dst, std::uint64_t size)
{
    const auto gid = blockIdx.x * blockDim.x + threadIdx.x;

    if(gid < SDIV(size, ChunkSize))
    {
        #pragma unroll ChunkSize
        for(uint64_t i = 0; i < ChunkSize; i++)
        {
            dst[gid*ChunkSize+i] = src[gid*ChunkSize+i];
        }  
    }
}

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

    static constexpr std::uint64_t chunk_size = 8;

    // memcpy bandwidth
    cudaSetDevice(src_id); CUERR
    for(auto size : sizes)
    {
        std::cout << size << " bytes" << std::endl;
        for(index_t i = 0; i < repeats; i++)
        {
            cudaEventRecord(start);
            cudaMemcpy(src_ptr, dst_ptr, sizeof(data_t)*size, D2D);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&(timings[i]), start, stop); CUERR
        }
        std::cout << "host memcpy" << std::endl;
        print_csv(timings.begin(), timings.end());
        print_statistic(timings.begin(), timings.end());

        /*
        for(index_t i = 0; i < repeats; i++)
        {
            cudaEventRecord(start);
            memcpy_kernel<<<1, 1>>>(src_ptr, dst_ptr, size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&(timings[i]), start, stop); CUERR
        }
        std::cout << "device memcpy" << std::endl;
        print_csv(timings.begin(), timings.end());
        print_statistic(timings.begin(), timings.end());
        */

        for(index_t i = 0; i < repeats; i++)
        {
            cudaEventRecord(start);
            memcpy2_kernel<data_t, chunk_size>
            <<<SDIV(SDIV(size, chunk_size), 1024), 1024>>>(src_ptr, dst_ptr, size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&(timings[i]), start, stop); CUERR
        }
        std::cout << "threaded device memcpy" << std::endl;
        print_csv(timings.begin(), timings.end());
        print_statistic(timings.begin(), timings.end());

        std::cout << std::endl;
    }

    // cleanup
    cudaSetDevice(src_id); CUERR
    cudaEventDestroy(start); CUERR
    cudaEventDestroy(stop); CUERR
    cudaFree(src_ptr); CUERR

    cudaSetDevice(dst_id); CUERR
    cudaFree(dst_ptr); CUERR
}
#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <string>

/////////////////
// Image Class //
///////////////// 
class Image{
    public:
        float* data = nullptr;
        unsigned int width, height;
        
        Image(std::string filename);
        Image(float* data, unsigned int w, unsigned int h);
        ~Image();
        void print();
        void save(std::string outputfilename);
        bool equals(Image* img);
        size_t get_size();

        static Image* averaging_filter(unsigned int dim);

};

struct cudaTime_t
{
    cudaEvent_t start, stop;
    void start_time();
    void stop_time(float& time);
};

struct performance_t
{
    float runtime;
    float throughput;
    float bandwidth;
    void clear();
};

enum MemType
{
    GlobalMemory,
    SharedMemory,
    TextureMemory
};

std::string get_string_from_enum(MemType t);

#endif
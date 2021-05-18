#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <string>

// Just renamed the datatypes for no reason at all.
typedef float pixel_t;
typedef pixel_t* image_t;
typedef unsigned int dim_t;

void imread(std::string filename, image_t* image, dim_t& width, dim_t& height);
void imsave(std::string filename, image_t image, dim_t& width, dim_t& height);

/**
* Generates a kernel of dimention @param:dim and stores in in @param:kernel
*/
void averaging_filter(image_t kernel, dim_t& dim);

struct cudaTime_t{
    cudaEvent_t start, stop;
    void start_time();
    void stop_time(float& time);
};

struct performance_t {
    float runtime;
    float throughput;
    float bandwidth;
    void clear();
};

#endif
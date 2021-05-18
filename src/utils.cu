#include <string>
#include <cuda_runtime.h>
#include "utils.h"
#include "helper_functions.h"
#include "helper_cuda.h"   

#define OUTDIR "output/"

void imread(std::string filename, image_t* image, dim_t& width, dim_t& height){

    char *imagePath = sdkFindFilePath(filename.c_str(), "convolution");

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", filename.c_str());
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, image, &width, &height);
    // printf("Loaded Image %s with %d, %d\n", imagePath, width, height);
    free(imagePath);
    
}

void imsave(std::string filename, image_t image, dim_t& width, dim_t& height){
    filename = OUTDIR+filename;
    sdkSavePGM(filename.c_str(), image, width, height);
    // printf("Wrote '%s'\n", filename.c_str());
}


void averaging_filter(image_t kernel, dim_t& dim){
    int size = dim*dim;
    pixel_t p = 1.0/size;
    for(int i = 0;i<size;i++) kernel[i] = p;
}


void cudaTime_t::start_time(){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
}


void cudaTime_t::stop_time(float& time){
    cudaEventRecord(stop,0);
    cudaEventSynchronize( stop );
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );
}

void performance_t::clear(){
    runtime = 0;
    throughput = 0;
    bandwidth = 0;
}

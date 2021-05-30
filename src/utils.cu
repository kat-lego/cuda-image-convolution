#include <string>
#include <cuda_runtime.h>
#include "utils.h"
#include "helper_functions.h"
#include "helper_cuda.h"   

#define OUTDIR "output/"

using namespace std;

/////////////////
// Image Class //
///////////////// 

Image::Image(string imagefilename)
{
    char *imagePath = sdkFindFilePath(imagefilename.c_str(), "convolution");

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imagefilename.c_str());
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &data, &width, &height);
    printf("Loaded '%s', %d x %d pixels\n", imagefilename.c_str(), width, height);
}

Image::Image(float* data, unsigned int w, unsigned int h)
{
    this->data = data;
    this->width = w;
    this->height = h;
}


Image::~Image()
{
    delete[] data;
    printf("Data Freed\n");
}

void Image::print()
{
    for(int row=0; row<height; row++)
    {
        for(int col=0; col<width; col++)
        {
            printf("%0.2f ",data[row*width+col]);
        }
        printf("\n");
    }
}

void Image::save(string outputfilename)
{
    outputfilename = OUTDIR+outputfilename;
    sdkSavePGM(outputfilename.c_str(), data, width, height);
    printf("Wrote '%s'\n", outputfilename.c_str());
}

bool Image::equals(Image* img)
{
    if(img->width!=this->width || img->height!=this->height){
        printf("lengths are different: %u=!%u and %u!=%u\n", img->width,this->width,img->height,this->height);
        return false;
    }
    bool res = true;
    for(int i=0;i<width;i++)
        for(int j=0;j<height;j++)
            if( abs(img->data[i*width+j] - this->data[i*width+j])>0.0001 )
                res = false;

    return res;
}

size_t Image::get_size(){
    return width*height*sizeof(float);
}


Image* Image::averaging_filter(unsigned int dim)
{
    int size = dim*dim;
    float* data = new float[size];

    float p = 1.0/size;
    for(int i = 0;i<size;i++) data[i] = p;

    return new Image(data, dim, dim);
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

std::string get_string_from_enum(MemType t)
{
    switch(t){
        case GlobalMemory:
            return "GlobalMemory";
        case SharedMemory:
            return "SharedMemory";
        case TextureMemory:
            return "TextureMemory";
        default:
            return "INVALID ENUM";
    }
}
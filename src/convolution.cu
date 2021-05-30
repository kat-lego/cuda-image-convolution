#include <string>
#include <stdio.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "helper_functions.h"
#include "helper_cuda.h"

#define MAX_FILTER_WIDTH 21
#define TILE_WIDTH 32
#define MAX_SM 52*52

__constant__ float const_filter[MAX_FILTER_WIDTH*MAX_FILTER_WIDTH];
__constant__ unsigned int const_filter_dim;

__global__ void cuda_convolution_globalmem_kernel(float* input, float* output, unsigned int width, unsigned int height);
__global__ void cuda_convolution_sharedmem_kernel(float* input, float* output, unsigned int width, unsigned int height);
__global__ void cuda_convolution_texturemem_kernel(float* output, unsigned int width, unsigned int height);

texture<float, 2, cudaReadModeElementType>tex_image;

///////////////////////////////////////////////
// A Serial Implementation of 2d convolution //
///////////////////////////////////////////////
Image* serial_convolution(Image* input_image, Image* filter)
{
    cudaTime_t timer;
    timer.start_time();
    
    int filter_dim = filter->width;

    float* output_data = new float[input_image->width*input_image->height];

    for(int row=0; row<input_image->height; row++){
        for(int col=0; col<input_image->width; col++){
                    
            float p = 0;
            int r,c, r1, c1;

            for(int i=0;i<filter_dim; i++){
                for(int j =0; j<filter_dim; j++){
        
                    r = row+i-filter_dim/2; c = col+j-filter_dim/2;
                    r1 = filter_dim-i-1; c1 = filter_dim-j-1;
                    if(r>=0 && r<input_image->height && c>=0 && c<input_image->width){
                        p += filter->data[r1*filter_dim+c1]*input_image->data[r*input_image->width+c];
                    }
                }
            }

            output_data[row*input_image->width+col] = p;
            
        }
    }

    float time;
    timer.stop_time(time);

    printf("Serial Convolution finished running after %f ms\n", time);

    return new Image(output_data, input_image->width, input_image->height);
}

//////////////////////
// Cuda Entry point //
//////////////////////
Image* cuda_convolution(Image* input_image, Image* filter, MemType memtype)
{
    //allocate device memory
    float* dev_input; float* dev_output;

    size_t size = input_image->get_size();
    checkCudaErrors( cudaMalloc( (void**)&dev_input,  size ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_output, size ) );

    checkCudaErrors
    ( 
        cudaMemcpy( dev_input, input_image->data, size , cudaMemcpyHostToDevice)
    );
    
    // fill in the filter on constant memory
    checkCudaErrors
    ( 
        cudaMemcpyToSymbol(const_filter_dim, &filter->width, sizeof(unsigned int))
    );
    checkCudaErrors
    (
        cudaMemcpyToSymbol(const_filter, filter->data, filter->get_size())
    );

    //preparing for texture memory
    if(memtype == TextureMemory)
    {
        cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        
        // Set texture parameters
        tex_image.addressMode[0] = cudaAddressModeWrap;
        tex_image.addressMode[1] = cudaAddressModeWrap;
        tex_image.filterMode = cudaFilterModeLinear;
        tex_image.normalized = false;

        // Bind the array to the texture
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        checkCudaErrors(
            cudaBindTexture2D( nullptr, tex_image, dev_input, desc, input_image->width, input_image->height, sizeof(float) * input_image->width )
        );

    }

    // figure out block dimensions
    int bw = (input_image->width+TILE_WIDTH-1)/TILE_WIDTH;
    int hw = (input_image->height+TILE_WIDTH-1)/TILE_WIDTH;

    dim3 grid_d(bw, hw);
    dim3 block_d(TILE_WIDTH, TILE_WIDTH);

    cudaTime_t timer;float time;
    timer.start_time();

    //call the appropriate kernel
    switch(memtype){
        case GlobalMemory:
            cuda_convolution_globalmem_kernel<<<grid_d, block_d>>>
                (dev_input, dev_output, input_image->width, input_image->height);
            break;
        case SharedMemory:
            cuda_convolution_sharedmem_kernel<<<grid_d, block_d>>>
                (dev_input, dev_output, input_image->width, input_image->height);
            break;
        case TextureMemory:
            cuda_convolution_texturemem_kernel<<<grid_d, block_d>>>
                (dev_output, input_image->width, input_image->height);
            break;
    }
    
    timer.stop_time(time);

    // copy output image back to host
    float* output_data = new float[input_image->width*input_image->height];
    checkCudaErrors( cudaMemcpy( output_data, dev_output, size, cudaMemcpyDeviceToHost) );
    
    checkCudaErrors( cudaFree(dev_input) );
    checkCudaErrors( cudaFree(dev_output) );

    checkCudaErrors( cudaUnbindTexture( tex_image ) );
    
    printf("Cuda Convolution with %s finished running after %f ms\n", get_string_from_enum(memtype).c_str(),time);

    return new Image(output_data, input_image->width, input_image->height);
}

///////////////////////////////////////////////////////////
// 2d convolution with global memory and constant memory //
///////////////////////////////////////////////////////////
__global__ void cuda_convolution_globalmem_kernel(float* input, float* output, unsigned int width, unsigned int height)
{
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int r,c,r1,c1;

    if(row>=height && col>=width) return;

    float p = 0;
    for(int i=0;i<const_filter_dim; i++){
        for(int j =0; j<const_filter_dim; j++){

            r = row+i-const_filter_dim/2; c = col+j-const_filter_dim/2;
            r1 = const_filter_dim-i-1; c1 = const_filter_dim-j-1;
            if(r>=0 && r<height && c>=0 && c<width){
                p += const_filter[r1*const_filter_dim+c1]*input[r*width+c];
            }
        }
    }

    output[row*width+col] = p;

}

///////////////////////////////////////////////////////////
// 2d convolution with shared memory and constant memory //
///////////////////////////////////////////////////////////
__global__ void cuda_convolution_sharedmem_kernel(float* input, float* output, unsigned int width, unsigned int height)
{
    __shared__ float ds[MAX_SM];
    int ds_width = TILE_WIDTH+const_filter_dim-1;

    int bx = blockIdx.x ; int by = blockIdx.y ;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    if(row>=height && col>=width) return;
    
    int rad = const_filter_dim/2;
    int r = by*TILE_WIDTH - rad; int c = bx*TILE_WIDTH - rad;
    int r1,c1;

    int i = ty*TILE_WIDTH+tx;

    while(i<ds_width*ds_width){
        r1 = r+i/ds_width; c1 = c+i%ds_width;
        if(r1>=0 && r1<height && c1>=0 && c1<width)
            ds[i]= input[r1*width+c1];
        else
            ds[i] = -1;
        i+=TILE_WIDTH*TILE_WIDTH;
    }
    __syncthreads();

    float p = 0;

    for(int i=0;i<const_filter_dim; i++){
        for(int j =0; j<const_filter_dim; j++){
            r1 = const_filter_dim-i-1; c1 = const_filter_dim-j-1;
            r = ty+i; c = tx+j;
            
            if(ds[r*ds_width+c]!=-1){
                p+=const_filter[r1*const_filter_dim+c1]*ds[r*ds_width+c];
            }
        }
    }

    output[row*width+col] =p;

}

////////////////////////////////////////
// 2d convolution with texture memory //
////////////////////////////////////////
__global__ void cuda_convolution_texturemem_kernel(float* output, unsigned int width, unsigned int height)
{
    int bx = blockIdx.x ; int by = blockIdx.y ;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;


    if(row>=height && col>=width) return;
    int r,c,r1,c1;

    float p = 0;
    for(int i=0;i<const_filter_dim; i++)
    {
        for(int j =0; j<const_filter_dim; j++)
        {

            r = row+i-const_filter_dim/2; c = col+j-const_filter_dim/2;
            r1 = const_filter_dim-i-1; c1 = const_filter_dim-j-1;
            float v = tex2D(tex_image, c, r);
            if(r>=0 && r<height && c>=0 && c<width){
                p += const_filter[r1*const_filter_dim+c1]*v;
            }
        }
    }
    
    output[row*width+col] = p;
}
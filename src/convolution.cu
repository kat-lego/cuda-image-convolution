#include <string>
#include <stdio.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "helper_functions.h"
#include "helper_cuda.h"

#define MAX_FILTER_WIDTH 21
#define TILE_WIDTH 32
#define MAX_SM 52*52

__constant__ pixel_t filter[MAX_FILTER_WIDTH*MAX_FILTER_WIDTH];
__constant__ dim_t filter_dim;

texture<pixel_t, 2, cudaReadModeElementType>tex_input;
texture<pixel_t>tex_filter;

__global__ void cuda_convolution_gc(image_t input, image_t output, dim_t width, dim_t height);
__global__ void cuda_convolution_sc(image_t input, image_t output, dim_t width, dim_t height);
__global__ void cuda_convolution_sc2(image_t input, image_t output, dim_t width, dim_t height);
__global__ void cuda_convolution_tc(image_t output, dim_t width, dim_t height);


/**
* a serial implementation of 2d convolution
*/
void serial_convolution(image_t input, image_t kernel, image_t output, dim_t& width, dim_t& height, dim_t& k_dim, performance_t& p){
    cudaTime_t time;
    time.start_time();
    
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
                    
            pixel_t p = 0;
            int radius = k_dim/2;
            int r,c, rk, ck;

            for(int a=0; a<k_dim;a++){
                for(int b=0; b<k_dim; b++){
                    r = i+a-radius;
                    c = j+b-radius;
                    
                    rk = k_dim-a-1;
                    ck = k_dim-b-1;
                    
                    if(r>=0 && r<height && c>=0 && c<width){
                        p += kernel[rk*k_dim + ck]*input[r*width+c];
                    }
                }
            }

            output[i*width+j] = p;
            
        }
    }

    time.stop_time(p.runtime);
    long N = 2*width*height*k_dim*k_dim;
    p.throughput = N / (p.runtime*1000000.0f);

}

void cuda_convolution(image_t input, image_t kernel, image_t output, dim_t& width, dim_t& height, dim_t& k_dim, int version, performance_t& p){
    
    //allocate device memory
    image_t dev_input, dev_output;

    size_t size = width*height*sizeof(dim_t);
    cudaMalloc( (void**)&dev_input,  size );
    cudaMalloc( (void**)&dev_output, size );

    cudaMemcpy( dev_input, input, size , cudaMemcpyHostToDevice);

    //fill in the filter on constant memory
    size_t k_size = k_dim*k_dim*sizeof(pixel_t);

    cudaMemcpyToSymbol(filter_dim, &k_dim, sizeof(dim_t));
    cudaMemcpyToSymbol(filter, kernel, k_size);

    image_t dev_filter;
    cudaMalloc( (void**)&dev_filter, k_size );
    cudaMemcpy( dev_filter, kernel, k_size , cudaMemcpyHostToDevice);

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<pixel_t>();
    cudaBindTexture2D( nullptr, tex_input, input, desc, width, height, sizeof(pixel_t) * width );
    
    cudaBindTexture( nullptr, tex_filter, dev_filter, k_size );

    //figure out block dimensions
    int bw = (width+TILE_WIDTH-1)/TILE_WIDTH;
    int hw = (height+TILE_WIDTH-1)/TILE_WIDTH;

    dim3 grid_d(bw, hw);
    dim3 block_d(TILE_WIDTH, TILE_WIDTH);

    cudaTime_t time;
    time.start_time();

    //call the appropriate kernel
    switch(version){
        case 1:
            cuda_convolution_gc<<<grid_d, block_d>>>(dev_input, dev_output, width, height);
            break;
        case 2:
            cuda_convolution_sc<<<grid_d, block_d>>>(dev_input, dev_output, width, height);
            break;
        case 3:
            cuda_convolution_tc<<<grid_d, block_d>>>(dev_output, width, height);
            break;
    }

    time.stop_time(p.runtime);
    long N = 2*width*height*k_dim*k_dim;
    p.throughput = N / (p.runtime*1000000.0f);

    // copy output image back to host
    cudaMemcpy( output, dev_output, size, cudaMemcpyDeviceToHost);
    
    //free memory
    cudaFree(dev_input);
    cudaFree(dev_output);
    
    cudaUnbindTexture( tex_input );
    cudaUnbindTexture( tex_filter );

}

/**
* 2d convolution with global memory and constant memory
*/
__global__ void cuda_convolution_gc(image_t input, image_t output, dim_t width, dim_t height){
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int r,c,r1,c1;

    if(row>=height && col>=width) return;

	if(bx==0 && by==0 && tx==0 && ty==0)
		printf("Hello Leo\n");
    pixel_t p = 0;
    for(int i=0;i<filter_dim; i++){
        for(int j =0; j<filter_dim; j++){

            r = row+i-filter_dim/2; c = col+j-filter_dim/2;
            r1 = filter_dim-i-1; c1 = filter_dim-j-1;
            if(r>=0 && r<height && c>=0 && c<width){
                p += filter[r1*filter_dim+c1]*input[r*width+c];
            }
        }
    }

    output[row*width+col] = p;
}

/**
* 2d convolution with shared memory and constant memory (just loading in the inner pixels)
*/
__global__ void cuda_convolution_sc2(image_t input, image_t output, dim_t width, dim_t height){
    __shared__ pixel_t ds[TILE_WIDTH][TILE_WIDTH];
    int ds_width = TILE_WIDTH;
    
    int bx = blockIdx.x ; int by = blockIdx.y ;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    if(row>=height && col>=width) return;

    ds[ty][tx] = input[row*width+col];

    __syncthreads();
    
    int r,c;
    int r1, c1;
    int rad = filter_dim/2;
    pixel_t p = 0;

    for(int i=0;i<filter_dim; i++){
        for(int j =0; j<filter_dim; j++){

            r1 = filter_dim-i-1; c1 = filter_dim-j-1;
            r = ty+i-rad; c = tx+j-rad;
            
            if(r>=0 && r<ds_width && c>=0 && c<ds_width){
                p+=filter[r1*filter_dim+c1]*ds[r][c];
            }else{
                r = row+i-rad; c = col+j-rad;
                if(r>=0 && r<height && c>=0 && c<width){
                    p += filter[r1*filter_dim+c1]*input[r*width+c];
                }
            }
        }
        
    }
    
    output[row*width+col] = p;

}


/**
* 2d convolution with shared memory and constant memory (loading in the inner pixels and halo)
*/
__global__ void cuda_convolution_sc(image_t input, image_t output, dim_t width, dim_t height){
    __shared__ pixel_t ds[MAX_SM];
    int ds_width = TILE_WIDTH+filter_dim-1;

    int bx = blockIdx.x ; int by = blockIdx.y ;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    if(row>=height && col>=width) return;
    
    int rad = filter_dim/2;
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

    pixel_t p = 0;

    for(int i=0;i<filter_dim; i++){
        for(int j =0; j<filter_dim; j++){
            r1 = filter_dim-i-1; c1 = filter_dim-j-1;
            r = ty+i; c = tx+j;
            
            if(ds[r*ds_width+c]!=-1){
                p+=filter[r1*filter_dim+c1]*ds[r*ds_width+c];
            }
        }
    }

    output[row*width+col] =p;

}


/**
* 2d convolution with texture memory
*/
__global__ void cuda_convolution_tc(image_t output, dim_t width, dim_t height){
    int bx = blockIdx.x ; int by = blockIdx.y ;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int r,c,r1,c1;

    if(row>=height && col>=width) return;

    pixel_t p = 0;
    for(int i=0;i<filter_dim; i++){
        for(int j =0; j<filter_dim; j++){

            r = row+i-filter_dim/2; c = col+j-filter_dim/2;
            r1 = filter_dim-i-1; c1 = filter_dim-j-1;
            pixel_t v = tex2D(tex_input, c, r);
            if(v != 0){
                p += tex1Dfetch(tex_filter, r1*filter_dim+c1 )*v;
            }
        }
    }

    output[row*width+col] = p;
}

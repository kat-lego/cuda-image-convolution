#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string>
#include <stdio.h>

#include "utils.h"
#include "helper_functions.h"
#include "helper_cuda.h"
#include "convolution.h"

using namespace std;

#define NUM_IMGS 4

static string image_files[] = {
    "input_images/mandrill128p.pgm", 
    "input_images/peppers256p.pgm",
    "input_images/lena512p.pgm",
    "input_images/man1024p.pgm"
};

bool check(image_t img1, image_t img2, int w, int h){
    bool res = true;

    for(int i=0;i<w;i++)
        for(int j=0;j<h;j++)
            if( abs(img1[i*w+j] - img2[i*w+j])>0.0001 )
                res = false;

    return res;
}


int main( int argc, char **argv ) {

    //Setup
    performance_t cuda_p,serial_p;
    cuda_p.clear();serial_p.clear();

    image_t input, output1, output2, filter;
    input = nullptr;
    
    dim_t width, height;

    imread(image_files[3], &input, width, height);
    output1 = new pixel_t[width*height];
    output2 = new pixel_t[width*height];

    dim_t dim = 21;
    filter = new pixel_t[dim*dim];
    averaging_filter(filter, dim);


    //Act
    serial_convolution(input, filter, output1, width, height, dim, serial_p);
    imsave("serial", output1, width, height);
    cout<< "Runtime: "<< serial_p.runtime << ", Throughput: "<< serial_p.throughput<<endl;
    
    cuda_convolution(input, filter, output2, width, height, dim, 1, cuda_p);
    imsave("cuda-global-memory", output2, width, height);
    cout<< "Runtime: "<< cuda_p.runtime << ", Throughput: "<< cuda_p.throughput<<", Check: "<<check(output1, output2, width,height)<<endl;

    cuda_convolution(input, filter, output2, width, height, dim, 2, cuda_p);
    imsave("cuda-shared-memory", output2, width, height);
    cout<< "Runtime: "<< cuda_p.runtime << ", Throughput: "<< cuda_p.throughput<<", Check: "<<check(output1, output2, width,height)<<endl;

    cuda_convolution(input, filter, output2, width, height, dim, 3, cuda_p);
    imsave("cuda-texured-memory", output2, width, height);
    cout<< "Runtime: "<< cuda_p.runtime << ", Throughput: "<< cuda_p.throughput<<", Check: "<<check(output1, output2, width,height)<<endl;

    //Tear down
    free(input);
    free(output1);
    free(output2);
    free(filter);

    return 0;
}

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


int main( int argc, char **argv ) {

    Image image(image_files[3]);
    Image* filter = Image::averaging_filter(7);

    Image* output_serial = serial_convolution(&image, filter);
    output_serial->save("serial");

    Image* output_cuda_g = cuda_convolution(&image, filter, GlobalMemory);
    output_cuda_g->save("cuda_global");
    printf("Cuda G Check: %d\n", output_serial->equals(output_cuda_g));

    Image* output_cuda_s = cuda_convolution(&image, filter, SharedMemory);
    output_cuda_s->save("cuda_shared");
    printf("Cuda S Check: %d\n", output_serial->equals(output_cuda_s));

    Image* output_cuda_t = cuda_convolution(&image, filter, TextureMemory);
    output_cuda_t->save("cuda_texture");
    printf("Cuda T Check: %d\n", output_serial->equals(output_cuda_t));

    delete filter;
    delete output_serial;
    delete output_cuda_g;
    delete output_cuda_s;
    delete output_cuda_t;
    return 0;
}

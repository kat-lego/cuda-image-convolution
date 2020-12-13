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

static string image_files[] = { "mandrill128p.pgm", "peppers256p.pgm", "lena512p.pgm", "man1024p.pgm"};

void run_tests(int index){

    //setup some variables
    int K = 4;
    int trials = 10; // number of trials
    performance_t p[K], tuna;

    image_t input, output, filter;
    input = nullptr;

    dim_t width, height, dim;

    // set up test data
    imread(image_files[index], &input, width, height);
    output = new pixel_t[width*height];

    printf("running tests on image \033[01;33m%s\033[0m with \033[01;33m%d\033[0m pixels*\n", image_files[index].c_str(), width*height);

    //setup output files
    string sufix[] = { "_serial.txt", "_cuda_gc.txt", "_cuda_sc.txt", "_cuda_tc.txt"};
    ofstream fout[K];
    for(int k=0;k<K;k++)
        fout[k].open( "../resfiles/"+to_string(index)+sufix[k]);

    for(dim = 9; dim<=21; dim+=2){
        filter = new pixel_t[dim*dim];
        averaging_filter(filter, dim);

        for(int k=0;k<K;k++)p[k].clear();

        for(int j=0;j<trials;j++){
            tuna.clear();
            serial_convolution(input, filter, output, width, height, dim, tuna);

            p[0].runtime+=tuna.runtime;
            p[0].throughput+=tuna.throughput;

            for(int k=1; k<=3; k++){
                tuna.clear();
                cuda_convolution(input, filter, output, width, height, dim, k, tuna);
                p[k].runtime+=tuna.runtime;
                p[k].throughput+=tuna.throughput;

            }

        }


        for(int k=0;k<4;k++){
            p[k].runtime/=trials;
            p[k].throughput/=trials;

            fout[k]<< dim << ", "<< p[k].runtime << ", "<< p[k].throughput<<endl;
        }


        free(filter);
    }
    printf("\033[1;32mtest completed, check resfiles\033[0m\n");

    //free stuff when done
    free(input);
    free(output);

    for(int k=0;k<K;k++)fout[k].close();


}

void run_validity_test(int type){

    performance_t tuna;
    image_t input, o1,o2,o3;
    input = nullptr;
    dim_t width, height;

    // set up test data
    imread(image_files[2], &input, width, height);

    o1 = new pixel_t[width*height];
    o2 = new pixel_t[width*height];
    o3 = new pixel_t[width*height];

    //filters
    dim_t d = 9;
    dim_t d2 = 3;

    //averaging filter
    pixel_t f1[d*d];
    averaging_filter(f1, d);

    //shapening filter
    pixel_t f2[] = {-1.0, -1.0, -1.0, -1.0, 9.0, -1.0, -1.0, -1.0, -1.0};

    //edge detection filter
    pixel_t f3[] = {-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0};

    if(type==0){
        serial_convolution(input, f1, o1, width, height, d, tuna);
        serial_convolution(input, f2, o2, width, height, d2, tuna);
        serial_convolution(input, f3, o3, width, height, d2, tuna);
    }else{
        cuda_convolution(input, f1, o1, width, height, d, type, tuna);
        cuda_convolution(input, f2, o2, width, height, d2, type, tuna);
        cuda_convolution(input, f3, o3, width, height, d2, type, tuna);
    }

    string outfile = to_string(type)+"_blur.pgm";
    imsave(outfile, o1, width, height);
    printf("\033[1;32msee file %s\033[0m\n", outfile.c_str());

    outfile = to_string(type)+"_sharp.pgm";
    imsave(outfile, o2, width, height);
    printf("\033[1;32msee file %s\033[0m\n", outfile.c_str());

    outfile = to_string(type)+"_edges.pgm";
    imsave(outfile, o3, width, height);
    printf("\033[1;32msee file %s\033[0m\n", outfile.c_str());

    free(input);
    free(o1);
    free(o2);
    free(o3);
}


int main( int argc, char **argv ) {
    printf("\033[01;33m+------------------------------Select an Option----------------------------+\n");
    printf("| 1- run performance tests                                                 |\n");
    printf("| 2- run validity tests (serial convolution)                               |\n");
    printf("| 3- run validity tests (cuda convolution with global and constant memory) |\n");
    printf("| 4- run validity tests (cuda convolution with shared and constant memory) |\n");
    printf("| 5- run validity tests (cuda convolution with texture memory)             |\n");
    printf("| -1 to exit                                                               |\n");
    printf("+--------------------------------------------------------------------------+\033[0m\n");


    printf("\n\033[0;34m>>>\033[0m ");
    int choice;
    cin>>choice;

    while(choice!=-1){
        if(choice==1){
            for(int i=0;i<NUM_IMGS;i++)run_tests(i);
            return 0;
        }else if(choice<=5 && choice>1){
            run_validity_test(choice-2);
        }

        printf("\n\033[0;34m>>>\033[0m ");
        cin>>choice;
    }


    return 0;
}

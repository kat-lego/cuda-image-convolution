#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <string>
#include "utils.h"


void serial_convolution(image_t input, image_t kernel, image_t output, dim_t& width, dim_t& height, dim_t& k_dim, performance_t& p);
void cuda_convolution(image_t input, image_t kernel, image_t output, dim_t& width, dim_t& height, dim_t& k_dim, int version, performance_t& p);

#endif
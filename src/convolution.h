#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <string>
#include "utils.h"


Image* serial_convolution(Image* input_image, Image* filter);
Image* cuda_convolution(Image* input_image, Image* filter, MemType memtype);

#endif
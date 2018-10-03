#ifndef __layers_opencl__
#define __layers_opencl__ 1
#include "utils.h"
#include "activations.h"

fm_t* convolve(conv_t* conv, fm_t* fm_in, int strides);
fm_t* connect(dense_t* dense, fm_t* fm_in);
fm_t* avg_pool(fm_t* fm_in);
// could be made in place?????
fm_t* normalize(bn_t* bn, fm_t* fm_in); 
fm_t* activate(fm_t* fm_in, activation_t activ);
fm_t* divide(fm_t* fm_in);
fm_t* add(fm_t* fm_in1, fm_t* fm_in2);
#endif
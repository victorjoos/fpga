#ifndef __layers_opencl__
#define __layers_opencl__ 1
#include "utils.h"
#include "activations.h"
#include "cl_utils.h"
// typedef enum layer {CONV, DENSE, AVG_POOL, BN, ACT, DIV, ADD} layer_type_t;
// typedef struct layer{
//     layer_type_t type;
//     void* layer;
// } layer_t;


fm_t* convolve(conv_t* conv, fm_t* fm_in, int strides, int first, cl_kernel* kernels);
fm_t* fully_connect(dense_t* dense, fm_t* fm_in);
fm_t* avg_pool(fm_t* fm_in);
// are in place!!
fm_t* normalize(bn_t* bn, fm_t* fm_in, int first); 
fm_t* activate(fm_t* fm_in, activation_t activ);
fm_t* divide(fm_t* fm_in);
fm_t* add(fm_t* fm_in1, fm_t* fm_in2, cl_kernel kernel);
#endif
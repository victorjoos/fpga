#ifndef __activations_opencl__
#define __activations_opencl__ 1

typedef enum activation {RELU, LEAKYRELU, TANH, BINARY, TERNARY} activation_t;
float act_relu(float x);
float leaky_relu(float x);
float act_tanh(float x);
float bin_htanh(float x);
float ter_htanh(float x);

#endif
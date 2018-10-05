#ifndef __activations_opencl__
#define __activations_opencl__ 1

typedef enum activation {RELU, LEAKYRELU, BINARY} activation_t;
float act_relu(float x);
float leaky_relu(float x);



#endif
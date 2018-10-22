#pragma once
#include <CL/cl.h>

typedef struct opencl_space{
    cl_context context; 
    cl_command_queue queue;
    cl_program program;
    cl_mem conv_kernel;
    cl_mem conv_bias;
    cl_mem fm_in;
    cl_mem fm_out;
}cl_space_t;
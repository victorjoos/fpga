#ifndef __opencl_My_clutils
#define __opencl_My_clutils 1

#include <CL/cl.h>
#include "utils.h"
typedef struct opencl_space{
    cl_context context; 
    cl_command_queue queue;
    cl_program program;
    cl_mem conv_kernel;
    cl_mem conv_bias;
    cl_mem fm_in;
    cl_mem fm_out;
}cl_space_t;


int init_cl(cl_space_t* space, char* file_name);
int load_kernel(char* kernel_name,
                    cl_space_t * space,
                    cl_kernel * kernel);
void free_cl(   cl_space_t* space,
                cl_kernel * kernel
                 );


cl_int cl_load_fm(fm_t* fm, cl_space_t* space);
cl_int cl_load_conv(conv_t* conv, cl_space_t* space);
cl_int cl_read_fm(fm_t* fm, cl_space_t* space);

#endif
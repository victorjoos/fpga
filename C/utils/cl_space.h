#pragma once
#include <CL/cl.h>

#define NMB_FM 5
typedef struct opencl_space{
    cl_context context; 
    cl_command_queue queue;
    cl_program program;
    cl_mem fm_fpga_buffers[NMB_FM];
    float * fm_buffers[NMB_FM];
    int act;
}cl_space_t;

#define MAX_FM_SIZE (32*32*64)
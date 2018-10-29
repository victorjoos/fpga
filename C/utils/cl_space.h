#pragma once
#include <CL/cl.h>

#define NMB_FM 5
typedef struct opencl_space{
    int taken[NMB_FM];
    float * fm_buffers[NMB_FM];
    cl_context context; 
    cl_command_queue queue;
    cl_command_queue mem_load_queue;
    cl_program program;
    cl_mem fm_fpga_buffers[NMB_FM];
    cl_command_queue mem_write_queue;
}cl_space_t;

#define MAX_FM_SIZE (32*32*16)
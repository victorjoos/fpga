
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "cl_utils.h"
#include "utils.h"
#define MAX_SOURCE_SIZE (0x100000)

int init_cl(cl_space_t* space, char* file_name){
    

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
            &device_id, &ret_num_devices);

    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen(file_name, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    // Create an OpenCL context
    space->context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    printf("%d\n", ret);

    // Create a command queue
    space->queue = clCreateCommandQueue(space->context, device_id, 0, &ret);

    // Create a program from the kernel source
    space->program = clCreateProgramWithSource(space->context, 1,
            (const char **)&source_str, (const size_t *)&source_size, &ret);
    // Build the program
    ret = clBuildProgram(space->program, 1, &device_id, NULL, NULL, NULL);
    size_t len = 0;
    ret = clGetProgramBuildInfo(space->program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char *buffer = calloc(len, sizeof(char));
    ret = clGetProgramBuildInfo(space->program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    printf("%d\nHello : \n%s\n", len, buffer);
    return 1;
}

int load_kernel(char* kernel_name,
                    cl_space_t * space,
                    cl_kernel * kernel){
    cl_int ret;
    // Load the kernel source code into the array source_str
    

    // create kernel
    *kernel = clCreateKernel(space->program, kernel_name, &ret);
    return 1;
}

void free_cl(   cl_space_t* space,
                cl_kernel * kernel
                 ){
    clReleaseMemObject(space->conv_kernel);
    clReleaseMemObject(space->conv_bias);
    clReleaseMemObject(space->fm_in);
    clReleaseMemObject(space->fm_out);
    cl_int ret;
    ret = clFlush(space->queue);
    ret = clFinish(space->queue);
    ret = clReleaseKernel(*kernel);
    ret = clReleaseProgram(space->program);
    ret = clReleaseCommandQueue(space->queue);
    ret = clReleaseContext(space->context);
}


cl_int cl_load_fm(fm_t* fm, cl_space_t* space){
    return clEnqueueWriteBuffer(space->queue, space->fm_in, CL_TRUE, 0,
            fm->nchannels*fm->fsize * sizeof(float), fm->values, 0, NULL, NULL);
}

cl_int cl_load_conv(conv_t* conv, cl_space_t* space){
    cl_int ret = clEnqueueWriteBuffer(space->queue, space->conv_kernel, CL_TRUE, 0,
            conv->size_in*conv->size_out*conv->xsize*conv->xsize * sizeof(float), 
            conv->kernel, 0, NULL, NULL);
    return clEnqueueWriteBuffer(space->queue, space->conv_bias, CL_TRUE, 0,
            conv->size_out * sizeof(float), 
            conv->bias, 0, NULL, NULL);
    
}

cl_int cl_read_fm(fm_t* fm, cl_space_t* space){
    return clEnqueueReadBuffer(space->queue, space->fm_out, CL_TRUE, 0,
            fm->nchannels*fm->fsize * sizeof(float), fm->values, 0, NULL, NULL);
}

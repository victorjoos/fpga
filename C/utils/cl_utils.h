#ifndef __opencl_My_clutils
#define __opencl_My_clutils 1

#include "cl_space.h"
#include <stdarg.h>
#include "utils.h"
extern cl_space_t *space;


int init_cl(char* file_name);
int load_kernel(char* kernel_name,
                    cl_kernel * kernel);
void free_cl(cl_kernel * kernel);


cl_int cl_load_fm(fm_t* fm);
void cl_load_conv(conv_t* conv);
void cl_read_fm(fm_t* fm);

// Print errors from PipeCNN
void printError(cl_int error);
void _checkError(int line, 
				const char *file, 
				cl_int error, 
				const char *msg, ...); // does not return
#define checkError(status, ...) _checkError(__LINE__, __FILE__, status, __VA_ARGS__)

#endif
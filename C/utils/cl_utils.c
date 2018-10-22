
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "cl_utils.h"
#include "utils.h"
#define MAX_SOURCE_SIZE (0x100000)

int init_cl(char* file_name){
    

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
            &device_id, &ret_num_devices);

	size_t name_size;
	ret = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, NULL, &name_size);
	char *name = malloc(name_size);
	ret = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, name_size, name, NULL);
	printf("platform: %s\n", name);
	free(name);
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &name_size);
	name = malloc(name_size);
	clGetDeviceInfo(device_id, CL_DEVICE_NAME, name_size, name, NULL);
	printf("device name: %s\n", name);
	free(name);
    FILE *fp;
    unsigned char *source_str;
    size_t source_size;

    fp = fopen(file_name, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
	fseek(fp, 0, SEEK_END);
  	source_size = ftell(fp);
	rewind(fp);
	printf("source size according to tell: %d\n", source_size);
    source_str = (unsigned char*)malloc(source_size);
    source_size = fread( source_str, 1, source_size, fp);
    fclose( fp );
	printf("source size: %d\n", source_size);

    // Create an OpenCL context
    space->context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    checkError(ret, "Failed create context");

    // Create a command queue
    space->queue = clCreateCommandQueue(space->context, device_id, 0, &ret);
    checkError(ret, "Failed create command queue");

    // Create a program from the kernel source
    /*space->program = clCreateProgramWithSource(space->context, 1,
            (const char **)&source_str, (const size_t *)&source_size, &ret);*/
	space->program = clCreateProgramWithBinary(space->context, 1, &device_id,  (const size_t*) &source_size, (const unsigned char**) &source_str, NULL, &ret);
    checkError(ret, "Failed create program with source");
	// Build the program
    ret = clBuildProgram(space->program, 1, &device_id, NULL, NULL, NULL);
    checkError(ret, "Failed build program");
    size_t len = 0;
    ret = clGetProgramBuildInfo(space->program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char *buffer = calloc(len, sizeof(char));
    ret = clGetProgramBuildInfo(space->program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
    printf("%d\nHello : \n%s\n", ret, buffer);
    return 1;
}

int load_kernel(char* kernel_name,
                    cl_kernel * kernel){
    cl_int ret;
    // Load the kernel source code into the array source_str
    

    // create kernel
    *kernel = clCreateKernel(space->program, kernel_name, &ret);
    checkError(ret, "Failed creating kernel");
    return 1;
}

void free_cl(cl_kernel * kernel){
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


cl_int cl_load_fm(fm_t* fm){
	// clSVMAlloc(space->context, CL_MEM_READ_WRITE,
    cl_int ret = clEnqueueWriteBuffer(space->queue, space->fm_in, CL_TRUE, 0,
            fm->nchannels*fm->fsize * sizeof(float), fm->values, 0, NULL, NULL);
    checkError(ret, "failed loading FMap");
}

void cl_load_conv(conv_t* conv){
    cl_int ret = clEnqueueWriteBuffer(space->queue, space->conv_kernel, CL_TRUE, 0,
            conv->size_in*conv->size_out*conv->xsize*conv->xsize * sizeof(float), 
            conv->kernel, 0, NULL, NULL);
    checkError(ret, "failed loading conv kernel");
    ret = clEnqueueWriteBuffer(space->queue, space->conv_bias, CL_TRUE, 0,
            conv->size_out * sizeof(float), 
            conv->bias, 0, NULL, NULL);
    checkError(ret, "failed loading bias");
}

void cl_read_fm(fm_t* fm){
    cl_int ret = clEnqueueReadBuffer(space->queue, space->fm_out, CL_TRUE, 0,
            fm->nchannels*fm->fsize * sizeof(float), fm->values, 0, NULL, NULL);
    checkError(ret, "failed reading fm");
}


void printError(cl_int error) {
	// Print error message
	switch(error)
	{
		case -1:
			printf("CL_DEVICE_NOT_FOUND ");
			break;
		case -2:
			printf("CL_DEVICE_NOT_AVAILABLE ");
			break;
		case -3:
			printf("CL_COMPILER_NOT_AVAILABLE ");
			break;
		case -4:
			printf("CL_MEM_OBJECT_ALLOCATION_FAILURE ");
			break;
		case -5:
			printf("CL_OUT_OF_RESOURCES ");
			break;
		case -6:
			printf("CL_OUT_OF_HOST_MEMORY ");
			break;
		case -7:
			printf("CL_PROFILING_INFO_NOT_AVAILABLE ");
			break;
		case -8:
			printf("CL_MEM_COPY_OVERLAP ");
			break;
		case -9:
			printf("CL_IMAGE_FORMAT_MISMATCH ");
			break;
		case -10:
			printf("CL_IMAGE_FORMAT_NOT_SUPPORTED ");
			break;
		case -11:
			printf("CL_BUILD_PROGRAM_FAILURE ");
			break;
		case -12:
			printf("CL_MAP_FAILURE ");
			break;
		case -13:
			printf("CL_MISALIGNED_SUB_BUFFER_OFFSET ");
			break;
		case -14:
			printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");
			break;

		case -30:
			printf("CL_INVALID_VALUE ");
			break;
		case -31:
			printf("CL_INVALID_DEVICE_TYPE ");
			break;
		case -32:
			printf("CL_INVALID_PLATFORM ");
			break;
		case -33:
			printf("CL_INVALID_DEVICE ");
			break;
		case -34:
			printf("CL_INVALID_CONTEXT ");
			break;
		case -35:
			printf("CL_INVALID_QUEUE_PROPERTIES ");
			break;
		case -36:
			printf("CL_INVALID_COMMAND_QUEUE ");
			break;
		case -37:
			printf("CL_INVALID_HOST_PTR ");
			break;
		case -38:
			printf("CL_INVALID_MEM_OBJECT ");
			break;
		case -39:
			printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");
			break;
		case -40:
			printf("CL_INVALID_IMAGE_SIZE ");
			break;
		case -41:
			printf("CL_INVALID_SAMPLER ");
			break;
		case -42:
			printf("CL_INVALID_BINARY ");
			break;
		case -43:
			printf("CL_INVALID_BUILD_OPTIONS ");
			break;
		case -44:
			printf("CL_INVALID_PROGRAM ");
			break;
		case -45:
			printf("CL_INVALID_PROGRAM_EXECUTABLE ");
			break;
		case -46:
			printf("CL_INVALID_KERNEL_NAME ");
			break;
		case -47:
			printf("CL_INVALID_KERNEL_DEFINITION ");
			break;
		case -48:
			printf("CL_INVALID_KERNEL ");
			break;
		case -49:
			printf("CL_INVALID_ARG_INDEX ");
			break;
		case -50:
			printf("CL_INVALID_ARG_VALUE ");
			break;
		case -51:
			printf("CL_INVALID_ARG_SIZE ");
			break;
		case -52:
			printf("CL_INVALID_KERNEL_ARGS ");
			break;
		case -53:
			printf("CL_INVALID_WORK_DIMENSION ");
			break;
		case -54:
			printf("CL_INVALID_WORK_GROUP_SIZE ");
			break;
		case -55:
			printf("CL_INVALID_WORK_ITEM_SIZE ");
			break;
		case -56:
			printf("CL_INVALID_GLOBAL_OFFSET ");
			break;
		case -57:
			printf("CL_INVALID_EVENT_WAIT_LIST ");
			break;
		case -58:
			printf("CL_INVALID_EVENT ");
			break;
		case -59:
			printf("CL_INVALID_OPERATION ");
			break;
		case -60:
			printf("CL_INVALID_GL_OBJECT ");
			break;
		case -61:
			printf("CL_INVALID_BUFFER_SIZE ");
			break;
		case -62:
			printf("CL_INVALID_MIP_LEVEL ");
			break;
		case -63:
			printf("CL_INVALID_GLOBAL_WORK_SIZE ");
			break;
		default:
			printf("UNRECOGNIZED ERROR CODE (%d)", error);
	}
}

// Print line, file name, and error code if there is an error. Exits the
// application upon error.
void _checkError(int line,
				const char *file,
				cl_int error,
                const char *msg,
                 ...) {
	// If not successful
	if(error != CL_SUCCESS) {
	// Print line and file
    printf("ERROR: ");
    printError(error);
    printf("\nLocation: %s:%d\n", file, line);

    // Print custom message.
    va_list vl;
    va_start(vl, msg);
    vprintf(msg, vl);
    printf("\n");
    va_end(vl);

    // Cleanup and bail.
    // free_cl();
    exit(error);
	}
}
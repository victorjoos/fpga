#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "utils.h"
#include "cl_utils.h"
#include "CL/cl.h"

const int IMSIZE = IMDIM*IMDIM*IMCHANNEL;
unsigned char* read_images(char* filename) {
    FILE *fileptr = fopen(filename, "rb");
    if (fileptr==NULL) {
        printf("no images\n");
        exit(-1);
    }
    size_t filelen = (IMSIZE + 1) * 10000; // image size + label
    unsigned char* buffer = (unsigned char*) malloc(filelen * sizeof(unsigned char));
    fread(buffer, sizeof(unsigned char), filelen, fileptr);
    fclose(fileptr);
    return buffer;
}

cl_mem alloc_shared_buffer (size_t size, float **host_ptr) {
  cl_int status;
  cl_mem device_ptr = clCreateBuffer(space->context, CL_MEM_ALLOC_HOST_PTR, sizeof(float) * size, NULL, &status);
  checkError(status, "Failed to create buffer");
  assert (host_ptr != NULL);
  *host_ptr = (float*) clEnqueueMapBuffer(space->queue, device_ptr, CL_TRUE, CL_MAP_WRITE|CL_MAP_READ, 0, sizeof(float) * size, 0, NULL, NULL, &status);
  checkError(status, "Failed to create shared pointer");
  assert (*host_ptr != NULL);
  return device_ptr;
}

/**
* - number: an image between 0 and 999
* - dataset : an array obtained from read_images
*/
unsigned char* get_image(int number, unsigned char* dataset) {
    return &dataset[(IMSIZE + 1)*number];
}
fm_t* img_to_fm(unsigned char* img){
    fm_t* fm = alloc_fm(3, 32);
    img += 1; // first elem is class
    for(int n=0; n<3; ++n){
        for(int i=0; i<fm->fdim; ++i){
            for(int j=0; j<fm->fdim; ++j){
                fm->values[n*32*32+i*32+j] = ((float)img[n*32*32+i*32+j])/255.0f;
            }
        }
    }
    return fm;
}


conv_t * read_conv(char* filename){
    
    conv_t * conv = (conv_t*) malloc(sizeof(conv_t));

    // read sizes
    FILE* fp = fopen(filename, "rb");
    if (fp==NULL) {
        printf("no dataset");
        exit(-1);
    }
    int sizes[3];
    fread(sizes, sizeof(int), 3, fp);
    conv->xsize = sizes[0];
    conv->size_in = sizes[1];
    conv->size_out = sizes[2];

    // read remaining values
    int kernel_size = conv->xsize*conv->xsize*conv->size_in*conv->size_out;

    float* values;
    conv->fpga_kernel = alloc_shared_buffer(kernel_size, &values);
    fread(values, sizeof(float), kernel_size, fp);
    conv->kernel = values;
    fclose(fp);
    return conv;
}
dense_t * read_dense(char* filename){
    dense_t * dense = (dense_t*) malloc(sizeof(dense_t));

    // read sizes
    FILE* fp = fopen(filename, "rb");
    if (fp==NULL) {
        printf("no dataset");
        exit(-1);
    }
    int sizes[2];
    fread(sizes, sizeof(int), 2, fp);
    dense->size_in = sizes[0];
    dense->size_out = sizes[1];
    
    // read remaining values
    int kernel_size = dense->size_in*dense->size_out;
    float* values;
    dense->fpga_kernel = alloc_shared_buffer(kernel_size, &values);
    fread(values, sizeof(float), kernel_size, fp);
    dense->kernel = values;
    fclose(fp);
    return dense;
}
bn_t * read_bn(char* filename){
    bn_t * bn = (bn_t*) malloc(sizeof(bn_t));

    // read sizes
    FILE* fp = fopen(filename, "rb");
    if (fp==NULL) {
        printf("no dataset");
        exit(-1);
    }
    int sizes[1];
    fread(sizes, sizeof(int), 1, fp);
    bn->size = sizes[0];
    
    // read remaining values
    float* values = (float*) malloc(sizeof(float) * 3*bn->size);
    fread(values, sizeof(float), 3*bn->size, fp);
    bn->beta = values;
    bn->mean = bn->beta + bn->size;
    bn->gamma = bn->mean + bn->size;
    fclose(fp);
    return bn;
}
fm_t* alloc_fm(int nchannels, int fdim){
    fm_t* fm = (fm_t*) malloc(sizeof(fm_t));
    fm->fdim = fdim; fm->fsize = fdim*fdim;
    fm->nchannels = nchannels;

    int act = 0;
    for(;act<NMB_FM && space->taken[act];act++);
    if(act>=NMB_FM) {
        printf("No Memory left\n");
        exit(-1);
    }
    space->taken[act] = 1;
    fm->fpga_values = space->fm_fpga_buffers[act];
    fm->values = space->fm_buffers[act];
    fm->mem_buff_channel = act;


    return fm;
}



// wrappers for matrices
float get_conv_elem(conv_t* conv, int k, int l, int inf, int outf){
    int zsize = conv->size_out;
    int ysize = zsize*conv->size_in;
    int xsize = ysize*conv->xsize;
    return conv->kernel[k*xsize + l*ysize + inf*zsize + outf];
}
void set_conv_elem(conv_t* conv, float value, int k, int l, int inf, int outf){
    int zsize = conv->size_out;
    int ysize = zsize*conv->size_in;
    int xsize = ysize*conv->xsize;
    conv->kernel[k*xsize + l*ysize + inf*zsize + outf] = value;
}

float get_dense_elem(dense_t * dense, int i, int j){
    return dense->kernel[i*dense->size_out + j];
}
float get_fm_elem(fm_t* fm, int channel, int i, int j){
    if((i<0)||(j<0)||(i>=fm->fdim)||(j>=fm->fdim)) return 0.0f;
    return fm->values[channel*fm->fsize + i*fm->fdim + j];
}
void set_fm_elem(fm_t* fm, float value, int channel, int i, int j){
    fm->values[channel*fm->fsize + i*fm->fdim + j] = value;
}

void print_fm(fm_t* fm, int n){
        printf("----Channel %d\n", n);
        for(int i=0; i<fm->fdim; ++i){
            for(int j=0; j<fm->fdim; ++j){
                printf("%3.2f ", get_fm_elem(fm, n, i, j));
            }
            printf("\n");
        }
}

void free_conv(conv_t* conv){
    // free(conv->kernel);
    clEnqueueUnmapMemObject (space->queue, conv->fpga_kernel, conv->kernel, 0, NULL, NULL);
    clReleaseMemObject (conv->fpga_kernel);
    free(conv);
}
void free_dense(dense_t* dense){
    // free(dense->kernel);
    clEnqueueUnmapMemObject (space->queue, dense->fpga_kernel, dense->kernel, 0, NULL, NULL);
    clReleaseMemObject (dense->fpga_kernel);
    free(dense);
}
void free_bn(bn_t* bn){
    free(bn->beta);
    free(bn);
}
void free_fm(fm_t* fm){
    // clEnqueueUnmapMemObject (space->queue, fm->fpga_values, fm->values, 0, NULL, NULL);
    // clReleaseMemObject (fm->fpga_values);
    free(fm);
    space->taken[fm->mem_buff_channel] = 0;
}

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "utils.h"
#include "activations.h"
#include "cl_utils.h"
#include "CL/cl.h"

const int IMSIZE = IMDIM*IMDIM*IMCHANNEL;
unsigned char* read_images(char* filename) {
    FILE *fileptr = fopen(filename, "rb");
    size_t filelen = (IMSIZE + 1) * 10000; // image size + label
    unsigned char* buffer = (unsigned char*) malloc(filelen * sizeof(unsigned char));
    fread(buffer, sizeof(unsigned char), filelen, fileptr);
    fclose(fileptr);
    return buffer;
}

cl_mem alloc_shared_buffer_uchar (size_t size, cl_uchar **host_ptr) {
  cl_int status;
  cl_mem device_ptr = clCreateBuffer(space->context, CL_MEM_ALLOC_HOST_PTR, sizeof(cl_uchar) * size, NULL, &status);
  checkError(status, "Failed to create buffer");
  assert (host_ptr != NULL);
  *host_ptr = (cl_uchar*) clEnqueueMapBuffer(space->queue[0], device_ptr, CL_TRUE, CL_MAP_WRITE|CL_MAP_READ, 0, sizeof(cl_uchar) * size, 0, NULL, NULL, &status);
  checkError(status, "Failed to create shared pointer");
  assert (*host_ptr != NULL);
  return device_ptr;
}
cl_mem alloc_shared_buffer_bn_vals (size_t size, bn_vals_t **host_ptr) {
  cl_int status;
  cl_mem device_ptr = clCreateBuffer(space->context, CL_MEM_ALLOC_HOST_PTR, sizeof(bn_vals_t) * size, NULL, &status);
  checkError(status, "Failed to create buffer");
  assert (host_ptr != NULL);
  *host_ptr = (cl_short*) clEnqueueMapBuffer(space->queue[0], device_ptr, CL_TRUE, CL_MAP_WRITE|CL_MAP_READ, 0, sizeof(cl_short) * size, 0, NULL, NULL, &status);
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
                fm->values[n*32*32+i*32+j] = (cl_short)img[n*32*32+i*32+j];
            }
        }
    }
    return fm;
}


conv_t * read_conv(char* filename){
    
    conv_t * conv = (conv_t*) malloc(sizeof(conv_t));

    // read sizes
    FILE* fp = fopen(filename, "rb");
    int sizes[3];
    fread(sizes, sizeof(int), 3, fp);
    conv->xsize = sizes[0];
    conv->size_in = sizes[1];
    conv->size_out = sizes[2];

    // read remaining values
    int kernel_size = conv->xsize*conv->xsize*conv->size_in*conv->size_out;

    cl_uchar* values;
    conv->fpga_kernel = alloc_shared_buffer_uchar(kernel_size, &values);

    float * _values = (float*)malloc(sizeof(float)*kernel_size);
    fread(_values, sizeof(float), kernel_size, fp);
    for(int i=0; i<kernel_size; ++i){
        cl_uchar nv = 0b00;
        if(_values[i]==0.f) nv = 0b10;
        else if(_values[i]>0.f) nv = 0b01;
        values[i] = nv;
    }
    conv->kernel = values;
    fclose(fp);
    free(_values);
    return conv;
}
dense_t * read_dense(char* filename){
    dense_t * dense = (dense_t*) malloc(sizeof(dense_t));

    // read sizes
    FILE* fp = fopen(filename, "rb");
    int sizes[2];
    fread(sizes, sizeof(int), 2, fp);
    dense->size_in = sizes[0];
    dense->size_out = sizes[1];
    
    // read remaining values
    int kernel_size = dense->size_in*dense->size_out;
    cl_uchar* values;
    dense->fpga_kernel = alloc_shared_buffer_uchar(kernel_size, &values);
    float * _values = (float*)malloc(sizeof(float)*kernel_size);
    fread(_values, sizeof(float), kernel_size, fp);
    for(int i=0; i<kernel_size; ++i){
        cl_uchar nv = 0b00;
        if(_values[i]==0.f) nv = 0b10;
        else if(_values[i]>0.f) nv = 0b01;
        values[i] = nv;
    }
    dense->kernel = values;
    free(_values);
    fclose(fp);
    return dense;
}
bn_t * read_bn(char* filename){
    bn_t * bn = (bn_t*) malloc(sizeof(bn_t));

    // read sizes
    FILE* fp = fopen(filename, "rb");
    int sizes[1];
    fread(sizes, sizeof(int), 1, fp);
    bn->size = sizes[0];
    
    // read remaining values
    float* _values = (float*) malloc(sizeof(float) * 2*bn->size);
    fread(_values, sizeof(float), 2*bn->size, fp);
    bn_vals_t* values;
    bn->fpga_values = alloc_shared_buffer_bn_vals(bn->size, &values);
    bn->values = values;
    for(int i=0; i<bn->size; ++i) {
        // TODO: add alert in case beta is too big
        values[i].beta = (cl_short) roundf(_values[i]*256.f);
    }
    for(int i=0, _i=bn->size; i<bn->size; ++i, ++_i) {
        // TODO: add alert in case gamma exponent is too big
        float l2 = log2f(fabsf(_values[_i]));
        l2 = roundf(l2);
        l2 = fminf(fmaxf(l2, -128.f), +127.f);
        // 8bits for exponent
        values[i].gamma = (cl_char) l2;
        // 8bits for sign (TODO: optimise??)
        values[i].gsign = (_values[_i]<0.f)? 1: 0;
    }
    fclose(fp);
    free(_values);
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
cl_uchar get_conv_elem(conv_t* conv, int k, int l, int inf, int outf){
    int zsize = conv->size_out;
    int ysize = zsize*conv->size_in;
    int xsize = ysize*conv->xsize;
    return conv->kernel[k*xsize + l*ysize + inf*zsize + outf];
}
void set_conv_elem(conv_t* conv, cl_uchar value, int k, int l, int inf, int outf){
    int zsize = conv->size_out;
    int ysize = zsize*conv->size_in;
    int xsize = ysize*conv->xsize;
    conv->kernel[k*xsize + l*ysize + inf*zsize + outf] = value;
}

cl_uchar get_dense_elem(dense_t * dense, int i, int j){
    return dense->kernel[i*dense->size_out + j];
}
cl_short get_fm_elem(fm_t* fm, int channel, int i, int j){
    if((i<0)||(j<0)||(i>=fm->fdim)||(j>=fm->fdim)) return 0;
    return fm->values[channel*fm->fsize + i*fm->fdim + j];
}
void set_fm_elem(fm_t* fm, cl_short value, int channel, int i, int j){
    fm->values[channel*fm->fsize + i*fm->fdim + j] = value;
}

void print_fm(fm_t* fm, int n, int fixed_point){
        printf("----Channel %d\n", n);
        for(int i=0; i<fm->fdim; ++i){
            for(int j=0; j<fm->fdim; ++j){
                cl_short fe = get_fm_elem(fm, n, i, j);
                if(fixed_point){
                    float ffe = (float)(fe>>8);
                    fe &= 0xff;
                    for(int i=1; i<9; ++i){
                        int on = (fe&0x80)>>7;
                        ffe += on*powf(2, -i);
                        fe = fe<<1;
                    }                
                    printf("%.3f ", ffe);
                } else {
                    printf("%d ", fe);
                }
            }
            printf("\n");
        }
}
void print_fm_sum(fm_t* fm, int fixed_point){
    for(int n=0;n<fm->nchannels; ++n){
        float ftotal = 0.f;
        int total = 0;
        for(int i=0; i<fm->fdim; ++i){
            for(int j=0; j<fm->fdim; ++j){
                cl_short fe = get_fm_elem(fm, n, i, j);
                if(fixed_point){
                    float ffe = (float)(fe>>8);
                    fe &= 0xff;
                    for(int i=1; i<9; ++i){
                        int on = (fe&0x80)>>7;
                        ffe += on*powf(2, -i);
                        fe = fe<<1;
                    }                
                    ftotal += ffe;
                } else {
                    total += fe;
                }
            }
        }
        if(fixed_point) printf("float cs: %.3f\n", ftotal);
        else            printf("int cs: %d\n", total);
    }
}

void free_conv(conv_t* conv){
    // free(conv->kernel);
    clEnqueueUnmapMemObject (space->queue[0], conv->fpga_kernel, conv->kernel, 0, NULL, NULL);
    clReleaseMemObject (conv->fpga_kernel);
    free(conv);
}
void free_dense(dense_t* dense){
    // free(dense->kernel);
    clEnqueueUnmapMemObject (space->queue[0], dense->fpga_kernel, dense->kernel, 0, NULL, NULL);
    clReleaseMemObject (dense->fpga_kernel);
    free(dense);
}
void free_bn(bn_t* bn){
    clEnqueueUnmapMemObject (space->queue[0], bn->fpga_values, bn->values, 0, NULL, NULL);
    clReleaseMemObject (bn->fpga_values);
    free(bn);
}
void free_fm(fm_t* fm){
    // clEnqueueUnmapMemObject (space->queue, fm->fpga_values, fm->values, 0, NULL, NULL);
    // clReleaseMemObject (fm->fpga_values);
    space->taken[fm->mem_buff_channel] = 0;
    free(fm);
}

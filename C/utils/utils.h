#ifndef __utility_opencl__
#define __utility_opencl__ 1
#define IMDIM 32
#define IMCHANNEL 3
#include "cl_space.h"
#include <CL/opencl.h>

// For the images
unsigned char* read_images(char* filename);
unsigned char* get_image(int number, unsigned char* dataset);

extern cl_space_t *space;

// For the layer's weights
typedef enum layer {CONV, BN, DENSE} layer_t;
typedef struct conv {
    float * kernel;
    float * bias;
    cl_mem fpga_kernel;
    cl_mem fpga_bias;
    int xsize;
    int size_in;
    int size_out;
} conv_t;

typedef struct dense {
    float * kernel;
    float * bias;
    cl_mem fpga_kernel;
    cl_mem fpga_bias;
    int size_in;
    int size_out;
} dense_t;

typedef struct bn {
    float * beta;
    float * mean;
    float * gamma;
    int size;
} bn_t;
typedef struct feature_map{
    float * values;
    cl_mem fpga_values;
    int nchannels;
    int fdim;
    int fsize; //= fdim*fdim
    int mem_buff_channel;
} fm_t;



conv_t * read_conv(char* filename);
dense_t * read_dense(char* filename);
bn_t * read_bn(char* filename);
fm_t* alloc_fm(int nchannels, int fdim);
fm_t* img_to_fm(unsigned char* img);

void print_fm(fm_t* fm, int n);


void free_conv(conv_t* conv);
void free_dense(dense_t* dense);
void free_bn(bn_t* bn);
void free_fm(fm_t* fm);

float get_conv_elem(conv_t* conv, int k, int l, int inf, int outf);
void set_conv_elem(conv_t* conv, float value, int k, int l, int inf, int outf);
float get_dense_elem(dense_t* dense, int i, int j);
float get_fm_elem(fm_t* fm, int channel, int i, int j);
void set_fm_elem(fm_t* fm, float value, int channel, int i, int j);
#endif
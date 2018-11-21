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
    cl_uchar * kernel;
    cl_mem fpga_kernel;
    int xsize;
    int size_in;
    int size_out;
} conv_t;

typedef struct dense {
    cl_uchar * kernel;
    cl_mem fpga_kernel;
    int size_in;
    int size_out;
} dense_t;

typedef struct bn_vals{
    cl_char gamma;
    cl_char gsign;
    cl_short beta;
} bn_vals_t;
typedef struct bn {
    bn_vals_t * values;
    cl_mem fpga_values;
    int size;
} bn_t;
typedef struct feature_map{
    cl_short * values;
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

void print_fm(fm_t* fm, int n, int fixed_point);
void print_fm_sum(fm_t* fm, int fixed_point);

void free_conv(conv_t* conv);
void free_dense(dense_t* dense);
void free_bn(bn_t* bn);
void free_fm(fm_t* fm);

cl_uchar get_conv_elem(conv_t* conv, int k, int l, int inf, int outf);
void set_conv_elem(conv_t* conv, cl_uchar value, int k, int l, int inf, int outf);
cl_uchar get_dense_elem(dense_t* dense, int i, int j);
cl_short get_fm_elem(fm_t* fm, int channel, int i, int j);
void set_fm_elem(fm_t* fm, cl_short value, int channel, int i, int j);
#endif
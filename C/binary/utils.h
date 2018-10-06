#ifndef __utility_opencl__
#define __utility_opencl__ 1
#define IMDIM 32
#define IMCHANNEL 3

// For the images
unsigned char* read_images(char* filename);
unsigned char* get_image(int number, unsigned char* dataset);

// For the layer's weights
typedef enum layer {CONV, BN, DENSE} layer_t;
typedef struct conv {
    int * kernel;
    float * bias;
    int xsize;
    int size_in;
    int size_out;
} conv_t;

typedef struct dense {
    int * kernel;
    float * bias;
    int size_in;
    int size_out;
} dense_t;

typedef struct bn {
    float * beta;
    float * gamma;
    float * mean;
    float * var;
    int size;
} bn_t;
typedef struct feature_map{
    int * values;
    int nchannels;
    int fdim;
    int fsize; //= fdim*fdim
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
void set_conv_elem(conv_t* conv, int value, int k, int l, int inf, int outf);
float get_dense_elem(dense_t* dense, int i, int j);
float get_fm_elem(fm_t* fm, int channel, int i, int j);
void set_fm_elem(fm_t* fm, int value, int channel, int i, int j);
#endif
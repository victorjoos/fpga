#ifndef __utility_opencl__
#define __utility_opencl__ 1
#define IMDIM 32
#define IMCHANNEL 3

// For the images
char* read_images(char* filename);
char* get_image(int number, char* dataset);

// For the layer's weights
typedef enum layer {CONV, BN, DENSE} layer_t;
typedef struct conv {
    float * kernel;
    float * bias;
    int strides;
    int size_in;
    int size_out;
} conv_t;

typedef struct dense {
    float * kernel;
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

conv_t * read_conv(char* filename);
dense_t * read_dense(char* filename);
bn_t * read_bn(char* filename);

float get_conv_elem(conv_t* conv, int i, int j, int k, int l);
float get_dense_elem(dense_t* dense, int i, int j);
#endif
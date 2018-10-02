#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

const int IMSIZE = IMDIM*IMDIM*IMCHANNEL;
char* read_images(char* filename) {
    FILE *fileptr = fopen(filename, "rb");
    size_t filelen = (IMSIZE + 1) * 10000; // image size + label
    char* buffer = (char*) malloc(filelen * sizeof(char));
    fread(buffer, sizeof(char), filelen, fileptr);
    fclose(fileptr);
    return buffer;
}

/**
* - number: an image between 0 and 999
* - dataset : an array obtained from read_images
*/
char* get_image(int number, char* dataset) {
    return &dataset[(IMSIZE + 1)*number];
}

conv_t * read_conv(char* filename){
    
    conv_t * conv = (conv_t*) malloc(sizeof(conv_t));

    // read sizes
    FILE* fp = fopen(filename, "rb");
    int sizes[3];
    fread(sizes, sizeof(int), 3, fp);
    conv->strides = sizes[0];
    conv->size_in = sizes[1];
    conv->size_out = sizes[2];
    printf("%d %d %d \n", conv->strides, conv->size_in, conv->size_out);
    // read remaining values
    int kernel_size = conv->strides*conv->strides*conv->size_in*conv->size_out;
    float* values = (float*) malloc(sizeof(float) * (kernel_size + conv->size_out));
    fread(values, sizeof(float), kernel_size+conv->size_out, fp);
    conv->kernel = values;
    conv->bias = values + kernel_size;
    fclose(fp);
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
    float* values = (float*) malloc(sizeof(float) * (kernel_size + dense->size_out));
    fread(values, sizeof(float), kernel_size+dense->size_out, fp);
    dense->kernel = values;
    dense->bias = values + kernel_size;
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
    float* values = (float*) malloc(sizeof(float) * 4*bn->size);
    fread(values, sizeof(float), 4*bn->size, fp);
    bn->beta = values;
    bn->gamma = bn->beta + bn->size;
    bn->mean = bn->gamma + bn->size;
    bn->var = bn->mean + bn->size;
    fclose(fp);
    return bn;
}



// wrappers for matrices
float get_conv_elem(conv_t* conv, int i, int j, int k, int l){
    int zsize = conv->size_out;
    int ysize = zsize*conv->size_in;
    int xsize = ysize*conv->strides;
    return conv->kernel[i*xsize + j*ysize + k*zsize + l];
}

float get_dense_elem(dense_t * dense, int i, int j){
    return dense->kernel[i*dense->size_out + j];
}




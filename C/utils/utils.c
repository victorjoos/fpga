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
fm_t* img_to_fm(char* img){
    fm_t* fm = alloc_fm(3, 32);
    img += 1; // first elem is class
    for(int i=0; i<3*fm->fsize; ++i) fm->values[i] = ((float)img[i])/255.0f;
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
    printf("%d %d %d \n", conv->xsize, conv->size_in, conv->size_out);
    // read remaining values
    int kernel_size = conv->xsize*conv->xsize*conv->size_in*conv->size_out;
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
fm_t* alloc_fm(int nchannels, int fdim){
    fm_t* fm = (fm_t*) malloc(sizeof(fm_t));
    fm->fdim = fdim; fm->fsize = fdim*fdim;
    fm->nchannels = nchannels;
    fm->values = (float*) malloc(sizeof(float) * nchannels*fm->fsize);
    return fm;
}



// wrappers for matrices
float get_conv_elem(conv_t* conv, int i, int j, int k, int l){
    int zsize = conv->size_out;
    int ysize = zsize*conv->size_in;
    int xsize = ysize*conv->xsize;
    return conv->kernel[i*xsize + j*ysize + k*zsize + l];
}

float get_dense_elem(dense_t * dense, int i, int j){
    return dense->kernel[i*dense->size_out + j];
}
float get_fm_elem(fm_t* fm, int channel, int i, int j){
    if((i<0)|(j<0)|(i>=fm->fdim)|(j>=fm->fdim)) return 0;
    return fm->values[channel*fm->fsize + i*fm->fdim + j];
}
void set_fm_elem(fm_t* fm, float value, int channel, int i, int j){
    fm->values[channel*fm->fsize + i*fm->fdim + j] = value;
}

void free_conv(conv_t* conv){
    free(conv->kernel);
    free(conv);
}
void free_dense(dense_t* dense){
    free(dense->kernel);
    free(dense);
}
void free_bn(bn_t* bn){
    free(bn->beta);
    free(bn);
}
void free_fm(fm_t* fm){
    free(fm->values); free(fm);
}


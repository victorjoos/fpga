#include <stdlib.h>
#include <stdio.h>
#include "../utils/utils.h"
#include "../utils/layers.h"
#include "layers.h"
conv_t* alloc_conv(int xsize, int size_in, int size_out){
    conv_t* conv = (conv_t*) malloc(sizeof(conv_t));
    conv->xsize = xsize; conv->size_in = size_in; conv->size_out = size_out;
    int kernel_size = conv->xsize*conv->xsize*conv->size_in*conv->size_out;
    float* values = (float*) malloc(sizeof(float) * (kernel_size + conv->size_out));
    conv->kernel = values;
    conv->bias = values + kernel_size;
    return conv;
}
void print_fm_test(fm_t* fm){
    for(int n=0; n<fm->nchannels; n++){
        printf("----Channel %d\n", n);
        for(int i=0; i<fm->fdim; ++i){
            for(int j=0; j<fm->fdim; ++j){
                printf("%3.2f ", get_fm_elem(fm, n, i, j));
            }
            printf("\n");
        }
    }
}

void simple_convolve(){
    conv_t* conv = alloc_conv(3, 3, 2);
    // set conv to 1 in center and 0 everywhere else and bias
    for(int outf=0; outf<conv->size_out; ++outf){
        for(int k=0; k<conv->xsize; ++k){
            for(int l=0; l<conv->xsize; ++l){
                for(int inf=0; inf<conv->size_in; ++inf){
                    set_conv_elem(conv, (k==l && k==1)? 1.0f: 0.0f, k, l, inf, outf);
                }
            }
        }
        conv->bias[outf] = 0.0f;
    }

    fm_t* fm = alloc_fm(3, 4); fm_t* fm2;
    float* vals = fm->values;
    for(int n=0; n<fm->nchannels; ++n){
        for(int i=0; i<fm->fsize; ++i){
            *vals = (float)(n+1);
            ++vals;
        }
    }

    // Test w/ strides = 1
    fm2 = convolve(conv, fm, 1); 
    print_fm_test(fm2); // todo everything = 6
    free_fm(fm2);
    
    // Test w/ strides = 2
    fm2 = convolve(conv, fm, 2);
    print_fm_test(fm2); // todo everything = 6
    free_fm(fm2);


    // set conv to 1 evrywhere except bias=0
    for(int outf=0; outf<conv->size_out; ++outf){
        for(int k=0; k<conv->xsize; ++k){
            for(int l=0; l<conv->xsize; ++l){
                for(int inf=0; inf<conv->size_in; ++inf){
                    set_conv_elem(conv, 1.0f, k, l, inf, outf);
                }
            }
        }
        conv->bias[outf] = 0.0f;
    }
    // Test w/ strides = 1
    fm2 = convolve(conv, fm, 1); 
    print_fm_test(fm2); // todo everything = 6
    free_fm(fm2);
    
    // Test w/ strides = 2
    fm2 = convolve(conv, fm, 2);
    print_fm_test(fm2); // todo everything = 6
    free_fm(fm2);


    // Free ressources
    free_fm(fm);
}

void random_convolution() {
    conv_t* conv = alloc_conv(3, 5, 2);
    for (int i=0; i<3*3*5*2; i++) {
        // set_conv_elem(conv, i, )
        conv->kernel[i] = i;
    }
    conv->bias[0] = 0;
    conv->bias[1] = 0;

    fm_t* fm = alloc_fm(5, 6); fm_t* fm2;
    for (int i=0; i<5*6*6; i++) {
        fm->values[i] = i;
    }
    fm2 = convolve(conv, fm, 1);
    print_fm_test(fm2);
}

void test_convolve(){
    
    //simple_convolve();
    random_convolution();
}

void test_connect(){

}

void test_activ(){

}

void test_normalize(){

}


void run_layer_tests(){
    test_convolve();
    test_connect();
    test_activ();
    test_normalize();
}
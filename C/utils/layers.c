#include "layers.h"
#include "utils.h"
#include "activations.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>




fm_t* convolve(conv_t* conv, fm_t* fm_in, int strides){
    assert(conv->size_in == fm_in->nchannels);
    fm_t* fm_out = alloc_fm(conv->size_out, fm_in->fdim/strides);
    const int offset = conv->xsize/2;
    for(int outf=0; outf<conv->size_out; ++outf){
        for(int _i=0; _i<fm_in->fdim; _i+=strides){
            for(int _j=0; _j<fm_in->fdim; _j+=strides){
                int i = _i-offset; int j = _j-offset;
                float acc = 0;
                for(int inf=0; inf<conv->size_in; ++inf){
                    for(int k=0; k<conv->xsize; ++k){
                        for(int l=0; l<conv->xsize; ++l){
                            acc += get_conv_elem(conv, k, l, inf, outf)*
                                get_fm_elem(fm_in, inf, i, j);
                        }
                    }
                }
                acc += conv->bias[outf];
                set_fm_elem(fm_out, acc, outf, _i, _j);
            }
        }
    }
    return fm_out;
}
fm_t* connect(dense_t* dense, fm_t* fm_in){
    assert(dense->size_in == fm_in->nchannels);
    fm_t* fm_out = alloc_fm(dense->size_out, 1);
    for(int outf=0; outf<dense->size_out; ++outf){
        for(int i=0; i<fm_in->fdim; ++i){
            for(int j=0; j<fm_in->fdim; ++j){
                float acc = 0;
                for(int inf=0; inf<dense->size_in; ++inf){
                    acc += get_dense_elem(dense, inf, outf)*
                        get_fm_elem(fm_in, inf, i, j);
                }
                acc += dense->bias[outf];
                set_fm_elem(fm_out, acc, outf, i, j);
            }
        }
    }
    return fm_out;
}
fm_t* avg_pool(fm_t* fm_in){
    fm_t* fm_out = alloc_fm(fm_in->nchannels, 1);
    float* channel = fm_in->values;
    for(int n=0; n<fm_in->nchannels; ++n){
        float acc = 0.0f;
        for(int i=0; i<fm_in->fsize; ++i){
            acc += *channel;
            ++channel;
        }
        set_fm_elem(fm_out, acc, n, 0, 0);
    }
    return fm_out;
}

fm_t* apply_f(fm_t* fm_in, float(*f)(float)){
    const int size = fm_in->nchannels*fm_in->fsize;
    for(int i=0; i<size; ++i)
        fm_in->values[i] = f(fm_in->values[i]);
    return fm_in;
}
fm_t* activate(fm_t* fm_in, activation_t activ){
    float (*f)(float);
    switch(activ){
        case RELU: f = act_relu; break;
        default:  f = act_relu;
    }
    return apply_f(fm_in, f);
}
float __div(float x) {return x*0.5f;}
fm_t* divide(fm_t* fm_in){
    return apply_f(fm_in, __div);
}
fm_t* add(fm_t* fm_in1, fm_t* fm_in2){
    assert(fm_in1->nchannels == fm_in2->nchannels);
    assert(fm_in1->fdim == fm_in2->fdim);
    const int size = fm_in1->nchannels*fm_in1->fsize;
    for(int i=0; i<size; ++i)
        fm_in1->values[i] += fm_in2->values[i];
    return fm_in1;
}
fm_t* normalize(bn_t* bn, fm_t* fm_in){
    assert(bn->size == fm_in->nchannels);
    float* values = fm_in->values;
    const float epsilon = 1e-3f;
    for(int n=0; n<fm_in->nchannels; ++n){
        float beta = bn->beta[n];
        float gamma = bn->gamma[n];
        float mean = bn->mean[n];
        float var = bn->var[n];
        for(int i=0; i<fm_in->fsize; ++i){
            float x = *values;
            *values = (x-mean) * gamma / sqrtf(var + epsilon) + beta;
            ++values;
        }
    }
    return fm_in;
}
#include "activations.h"
#include <math.h>
#include <stdio.h>
#include <CL/opencl.h>
float act_relu(float x){
    if(x < 0.0f) return 0.0f;
    return x;
}
float leaky_relu(float x){
    if(x < 0.0f) return 0.3f*x;
    return x;
}
float act_tanh(float x){
    return tanhf(x);
}
float bin_htanh(float x){
    float unclipped =(0.5f * x) + 0.5f;
    float clipped = fminf(fmaxf(unclipped, 0.f),  1.f);
    float plop = 2.f*round(clipped) - 1.f;
    return plop;
}

cl_short ter_htanh(cl_short x){
    cl_short ret;
    // if (fabsf(x)<0.5f) ret =  0.f;
    // else if (x<0)    ret = -1.f;
    // else             ret =  1.f;
    if(x<0){
        if( x >= -((cl_short)0x80) ) ret =  0;
        else         ret = -1;
    } else {
        if( x >= ((cl_short)0x80) ) ret =  1;
        else         ret =  0;
    }
    return ret;
}
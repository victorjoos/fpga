#include "activations.h"
#include <math.h>
#include <stdio.h>
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
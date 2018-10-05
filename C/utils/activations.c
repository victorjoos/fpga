#include "activations.h"
#include <math.h>
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
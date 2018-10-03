#include "activations.h"

float act_relu(float x){
    if(x < 0.0f) return 0.0f;
    return x;
}
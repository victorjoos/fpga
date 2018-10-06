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
int act_tanh(float x){
    x = (0.5f * x) + 0.5f;
    if (x<0.0f || x>1.0f) x = 0.0f;
    int new_x = (int) x;
    new_x = 2 * x - 1;
    return new_x;
}
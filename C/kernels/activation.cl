#include "../utils/utils.h"

__kernel activation_ff(__global const fm_t* fm) {
    int index = get_global_id(0);
    float x = fm->values[index];    
    if(x < 0.0f) {
        fm->values[index] = 0.3f*x;
    } else {
        fm->values[index] =  x;
    }
}
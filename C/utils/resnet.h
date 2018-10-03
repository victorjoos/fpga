#ifndef __resnet_opencl__
#define __resnet_opencl__ 1
#include "layers.h"
typedef struct resnet {
    int nblocks;
    conv_t** convs;
    dense_t** denses;
    bn_t** bns;
} resnet_t;

resnet_t* build_resnet(int nblocks, char* dir);
void free_resnet(resnet_t* resnet);
// returns the accuracy on the first img_sizes
double infer_resnet(resnet_t* resnet, char* imgs, int n_imgs);
#endif
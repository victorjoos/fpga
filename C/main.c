#include <stdlib.h>
#include <stdio.h>
// #include <hdf5.h>
#include "utils/utils.h"


int main(){
    char* dataset = read_images("../datasets/test_batch.bin");
    char* image_10 = get_image(1, dataset);
    printf("%hhx\n", image_10[0]);

    conv_t * conv = read_conv("../test/conv_1.bin");
    dense_t * dense = read_dense("../test/dense_1.bin");
    bn_t * bn = read_bn("../test/bn_1.bin");
    return 0;
}
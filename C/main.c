#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <ctype.h>
#include <getopt.h>
// #include <hdf5.h>
#include "utils/utils.h"
#include "utils/layers.h"
#include "utils/resnet.h"
#define CL_DEVICE_TYPE_CPU 2
void print_usage(){
    printf("Usage: ./main -d <directory> -s <resnet-size>\n");
}

int main( int argc, char * argv[]){
    /* Uncomment once code is ready
    int option = 0;
    char * dir = NULL;
    int nres = 0;
    while( (option = getopt(argc, argv, "d:s:")) != -1 ){
        switch(option){
            case 'd': dir = optarg; break;
            case 's': nres = atoi(optarg); break;
            default : printf("%d\n", option); print_usage(); exit(EXIT_FAILURE);
        }
    }
    if(dir == NULL | nres == 0) { print_usage(); exit(EXIT_FAILURE); }
    printf("%s %d\n", dir, nres);
    */

    unsigned char* dataset = read_images("../datasets/test_batch.bin");
    resnet_t* resnet = build_resnet(3, "../test2/");
    int n = 10;
    float acc = infer_resnet(resnet, dataset, n);
    printf("%.4f accuracy on %d images\n", acc, n);
    free_resnet(resnet);
    free(dataset);
    // char* image_10 = get_image(1, dataset);
    // printf("%hhx\n", image_10[0]);
    // conv_t * conv = read_conv("../test/conv_1.bin");
    // bn_t* bn = read_bn("../test/bn_1.bin");
    // fm_t * fm = activate(normalize(bn, convolve(conv, img_to_fm(image_10), 1)), RELU);
    // dense_t * dense = read_dense("../test/dense_1.bin");
    // bn_t * bn = read_bn("../test/bn_1.bin");
    return 0;
}
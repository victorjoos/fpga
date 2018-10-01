#include <stdlib.h>
#include <stdio.h>
#include <hdf5.h>
#define IMDIM 32
#define IMCHANNEL 3
const int IMSIZE = IMDIM*IMDIM*IMCHANNEL;
char* read_images(char* filename) {
    FILE *fileptr = fopen(filename, "rb");
    long filelen = (IMSIZE + 1) * 10000; // image size + label
    char* buffer = (char*) malloc(filelen * sizeof(char));
    fread(buffer, filelen, 1, fileptr);
    fclose(fileptr);
    return buffer;
}

/**
* - number: an image between 0 and 999
* - dataset : an array obtained from read_images
*/
char* get_image(int number, char* dataset) {
    return &dataset[(IMSIZE + 1)*number];
}

int main(){
    char* dataset = read_images("../datasets/test_batch.bin");
    char* image_10 = get_image(1, dataset);
    printf("%hhx\n", image_10[0]);
    return 0;
}
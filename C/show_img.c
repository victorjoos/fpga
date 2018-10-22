#include <stdlib.h>
#include <stdio.h>
#include <string.h>
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


int main_1(){
    char* dataset = read_images("../datasets/test_batch.bin");
    char* image_10 = get_image(55, dataset) + 1;


    int w = 32;
    int h = 32;
    FILE *f;
    unsigned char *img = NULL;
    int filesize = 54 + 3*w*h;  //w is your image width, h is image height, both int

    img = (unsigned char *)malloc(3*w*h);
    memset(img,0,3*w*h);

    int x,y,r,g,b;
    for(int i=0; i<w; i++)
    {
        for(int j=0; j<h; j++)
        {
            x=i; y=(h-1)-j;
            r = image_10[x+y*w];
            g = image_10[(32*32)+x+y*w];
            b = image_10[(32*32*2)+x+y*w];
            if (r > 255) r=255;
            if (g > 255) g=255;
            if (b > 255) b=255;
            img[(x+y*w)*3+2] = (unsigned char)(r);
            img[(x+y*w)*3+1] = (unsigned char)(g);
            img[(x+y*w)*3+0] = (unsigned char)(b);
        }
    }

    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};

    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(       w    );
    bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
    bmpinfoheader[ 6] = (unsigned char)(       w>>16);
    bmpinfoheader[ 7] = (unsigned char)(       w>>24);
    bmpinfoheader[ 8] = (unsigned char)(       h    );
    bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
    bmpinfoheader[10] = (unsigned char)(       h>>16);
    bmpinfoheader[11] = (unsigned char)(       h>>24);

    f = fopen("img.bmp","wb");
    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);
    for(int i=0; i<h; i++)
    {
        fwrite(img+(w*(h-i-1)*3),3,w,f);
        fwrite(bmppad,1,(4-(w*3)%4)%4,f);
    }

    free(img);
    fclose(f);
    return 0;
}
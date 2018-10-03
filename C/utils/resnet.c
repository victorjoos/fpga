#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "resnet.h"
#include "layers.h"
#include "utils.h"

char* assemble_name(char* prefix, int suffix, char* buffer){
    sprintf(buffer, "%d", suffix);
    strcat(prefix, buffer);
    strcat(prefix, ".bin");
    return prefix;
}

resnet_t* build_resnet(int nblocks, char* dir){
    const int n_bn = 6*nblocks+1;
    const int n_conv = n_bn + 2;
    const int n_dense = 1;
    resnet_t* resnet = (resnet_t*) malloc(sizeof(resnet_t));
    resnet->bns = (bn_t*) malloc(sizeof(bn_t)*n_bn + 
                    sizeof(conv_t)*n_conv + sizeof(dense_t)*n_dense);
    resnet->convs = (conv_t*) (resnet->bns + n_bn);
    resnet->denses = (dense_t*) (resnet->convs + n_conv);
    resnet->nblocks = nblocks;

    char buffer[40];
    for(int i=0; i<n_bn; ++i) {
        char bn_prefix[50] = ""; strcat(bn_prefix, dir); strcat(bn_prefix, "bn_");
        resnet->bns[i] = *(read_bn(assemble_name(bn_prefix, i+1, buffer)));
    }
    for(int i=0; i<n_conv; ++i) {
        char conv_prefix[50] = ""; strcat(conv_prefix, dir); strcat(conv_prefix, "conv_");
        resnet->convs[i] = *(read_conv(assemble_name(conv_prefix, i+1, buffer)));
    }
    for(int i=0; i<n_dense; ++i){
        char dense_prefix[50] = ""; strcat(dense_prefix, dir); strcat(dense_prefix, "dense_");
        resnet->denses[i] = *(read_dense(assemble_name(dense_prefix, i+1, buffer)));
    }
    
    return resnet;
}

double infer_resnet(resnet_t* resnet, char* imgs, int n_imgs){
    int ok = 0;
    const int n_stacks=3;
    for(int imgi=0; imgi<n_imgs; ++imgi){
        char* img = get_image(imgi, imgs);
        char img_class = img[0];
        fm_t* fm = img_to_fm(img);
        // TODO: free memory correctly
        
        // First non-residual block
        fm = convolve(resnet->convs, fm, 1);
        fm = normalize(resnet->bns, fm);
        fm = activate(fm, RELU);
        fm_t* fm_shortcut = fm;

        int conv_index = 1;
        int bn_index = 1;
        int short_conv_index = 0;
        // Loop over the 3 stacks, each containing nblocks
        for(int st=0; st<n_stacks; ++st){
            printf("stack %d\n", st);
            // Loop over the blocks
            for(int bl=0; bl<resnet->nblocks; ++bl){
                printf("block %d\n", bl);
                // Main block C->BN->Act->C->BN
                int strides = (st>0 && bl==0)? 2: 1;
                fm = convolve(resnet->convs + conv_index, fm, strides); ++conv_index;
                fm = normalize(resnet->bns + bn_index, fm); ++bn_index;
                fm = activate(fm, RELU);
                fm = convolve(resnet->convs + conv_index, fm, 1); ++conv_index;
                fm = normalize(resnet->bns + bn_index, fm); ++bn_index;
                
                // Update indices (not very beautyful but easier than recalculating)
                if (st>0 && bl==0) {
                    short_conv_index = conv_index;
                    ++conv_index;                    
                    // shortcut with dim reduction between stacks
                    fm_shortcut = convolve(resnet->convs + short_conv_index, 
                                            fm_shortcut, 2);
                }

                // Addition with shortcut
                fm = add(fm, fm_shortcut);
                fm = divide(fm);
                fm = activate(fm, RELU);

                // Update shortcut value
                fm_shortcut = fm;
                             
            }
        }
        fm = avg_pool(fm);
        fm = connect(resnet->denses, fm);
        float maxval = fm->values[0];
        char maxi = 0;
        for(int i=1; i<10; ++i){
            float fmv = fm->values[i];
            if(maxval<fmv){
                maxval = fmv;
                maxi = (char) i;
            }
        }
        printf("Expected %d and got %d\n", img_class, maxi);
        ok += (maxi==img_class);
    }

    return ((double) ok) / (double)n_imgs;
}
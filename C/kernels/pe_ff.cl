#include "config.h"

/*float get_fm_elem(__global const float* fm, int channel, int i, int j, int fdim, int fsize){
    if((i<0)||(j<0)||(i>=fdim)||(j>=fdim)) return 0.0f;
    return fm[channel*fsize + i*fdim + j];
}

void print_fm_test(__global const float* fm, int fdim, int fsize){
    int n=0;
    printf("----Channel %d x\n", n);
        for(int i=0; i<fdim; ++i){
            for(int j=0; j<fdim; ++j){
                printf("%.2f ", get_fm_elem(fm, n, i, j, fdim, fsize));
            }
            printf("\n");
        }
}*/


__kernel void pe_ff( const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const float* conv_kernel, __global const float* conv_bias,
                __global const float* fm_in, __global float* fm_out){
    const int outf = get_global_id(0);
    // printf("hello from conv\n");
    // conv consts
    const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize;         // TODO: avoid multiplication in kernel
    const int offset = ksize/2;
    
    // fm consts
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    
    for(int _i=(strides==2)?offset:0; _i<fdim_in; _i+=strides){
        for(int _j=(strides==2)?offset:0; _j<fdim_in; _j+=strides){
            int i = _i-offset; 
            int j = _j-offset;
            float acc = 0.0f;
            for(int inf=0; inf<conv_size_in; ++inf){
                for(int k=0; k<ksize; ++k){
                    for(int l=0; l<ksize; ++l){
                        float fm_elem;
                        if((i+k<0)||(j+l<0)||(i+k>=fdim_in)||(j+l>=fdim_in)) fm_elem = 0.0f;
                        else fm_elem = fm_in[inf*fsize_in + (i+k)*fdim_in + (j+l)];                                                
                        acc += conv_kernel[k*xsize + l*ysize + inf*zsize + outf] * fm_elem;
                    }
                }
            }
            acc += conv_bias[outf];
            fm_out[outf*fsize_out + _i/strides*fdim_out + _j/strides] = acc;
        }
    }
}

__kernel void pe_tile_ff( const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const float* conv_kernel, __global const float* conv_bias,
                __global const float* fm_in, __global float* fm_out){

    // conv consts
    const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize;         // TODO: avoid multiplication in kernel
    const int offset = ksize>>1;
    
    // fm consts
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    const int global_i = TILE_SIZE*get_group_id(0) + local_i;
    const int global_j = TILE_SIZE*get_group_id(1)+ local_j;
    // printf("%d %d\n", global_i, global_j);
    // if( get_group_id(0)==0) printf("%d %d\n", get_group_id(0), get_group_id(1));
    __local float fm_in_local[TILE_SIZE+2][TILE_SIZE+2];
    float acc;

    const int n_tiles = conv_size_out/TILE_SIZE;
    for(int outf=0; outf<conv_size_out; ++outf){
        for(int group_j=get_group_id(1); group_j<get_group_id(1)+1; ++group_j){
            acc = 0.0f;
            for(int inf=0; inf<conv_size_in; ++inf){
                // Load into shared memory
                int ii = TILE_SIZE*get_group_id(0) + 2*local_i - 1;
                int jj = TILE_SIZE*get_group_id(1) + 2*local_j - 1;
                for(int li=0; li<2; ++li){
                    for(int lj=0; lj<2; ++lj){
                        if(2*local_i+li>=TILE_SIZE) continue;
                        if(2*local_j+lj>=TILE_SIZE) continue;

                        float fm_elem;
                        if((ii+li<0)||(jj+lj<0)||(ii+li>=fdim_in)||(jj+lj>=fdim_in)) fm_elem = 0.0f;
                        else fm_elem = fm_in[inf*fsize_in + (ii+li)*fdim_in + (jj+lj)];
                        // printf("%d %d %.3f, \n", ii+li, jj+lj, fm_elem);
                        fm_in_local[2*local_i+li][2*local_j+lj] = fm_elem;
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                // if(get_local_id(0)==0 && get_local_id(1)==0){
                //     for(int x=0; x<TILE_SIZE+2;++x){
                //         for(int y=0; y<TILE_SIZE+2;++y){
                //             printf("%d ", fm_in_local[x][y]);
                //         }
                //         printf("\n");
                //     }
                // }

                int i = local_i; int j = local_j;
                for(int k=0; k<ksize; ++k){
                    for(int l=0; l<ksize; ++l){
                        float fm_elem = fm_in_local[i+k][j+l];                                              
                        acc += conv_kernel[k*xsize + l*ysize + inf*zsize + outf] * fm_elem;
                    }
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            acc += conv_bias[outf];
            // printf("%.3f\n", acc);
            fm_out[outf*fsize_out + global_i*fdim_out + global_j] = acc;
        }
    }
}


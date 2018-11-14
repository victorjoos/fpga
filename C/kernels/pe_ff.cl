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
#define TR 5 // use TR == TC ?
#define TC 5
#define TOUT 1
#define TIN 64
#define MAX_KSIZE 3

__kernel void pe_ff(const int conv_size_in, const int conv_size_out,
                    const int ksize, const int strides,
                    const int fdim_in, const int fdim_out,
                    __global const float* restrict conv_kernel,
                    __global const float* restrict fm_in,
                    __global float* restrict fm_out) {

    printf("[KERNEL] ksize: %d, strides: %d, conv_size_in: %d, conv_size_out: %d, fdim_in: %d, fdim_out: %d\n", ksize, strides, conv_size_in, conv_size_out, fdim_in, fdim_out);
    const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize; 
    const int offset = ksize/2;
    float l_out_fmap[TR][TC][TOUT];

    
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    const int max_conv_size = ksize*ksize*conv_size_in*conv_size_out;
    // printf("hello !: %d, %d, %d, %d\n", max_conv_size, ksize, conv_size_in, conv_size_out);
    for (int row=(strides==2)?0:-offset; row<fdim_in-offset; row += TR) {
        for (int col=(strides==2)?0:-offset; col<fdim_in-offset; col += TC) {
            for (int outf=0; outf<conv_size_out; outf += TOUT) {
                for (int inf=0; inf<conv_size_in; inf += TIN) {
                    // load memory here ...
                    float l_weights[MAX_KSIZE][MAX_KSIZE][TOUT][TIN];
                    float l_fmap[TR+MAX_KSIZE-1][TC+MAX_KSIZE-1][TIN];
                    // float l_out_fmap[TR][TC][TOUT];
                    // Load weights
                    for (int k=0; k<ksize; ++k) {
                        for (int l=0; l<ksize; ++l) {
                            for (int too=outf, _too=0; too<min(outf+TOUT, conv_size_out); ++too, ++_too) {
                                for (int tii=inf, _tii=0; tii<min(inf+TIN, conv_size_in); ++tii, ++_tii) {
                                    float fmap = conv_kernel[k*xsize + l*ysize + tii*zsize + too];
                                    l_weights[k][l][_too][_tii] = fmap;
                                }
                            }
                        }
                    }
                    // printf("inf: %d, outf: %d, row: %d, col: %d\n", inf, outf, row, col);
                    // for (int k=0; k<ksize; ++k) {
                    //     for (int l=0; l<ksize; ++l) {
                    //         printf("%.3f ", l_weights[k][l][outf][inf]);
                    //     }
                    //     printf("\n");
                    // }
                    // printf("\n");
                    // Load fmaps
                    for (int trr=row, _trr=0; trr<min(row+TR+ksize-1, fdim_in+offset); ++trr, ++_trr) {
                        for (int tcc=col, _tcc=0; tcc<min(col+TC+ksize-1, fdim_in+offset); ++tcc, ++_tcc) {
                            for (int tii=inf, _tii=0; tii<min(inf+TIN, conv_size_in); ++tii, ++_tii) {
                                float fm_elem;
                                if((trr<0)||(tcc<0)||(trr>=fdim_in)||(tcc>=fdim_in)) fm_elem = 0.0f;
                                else fm_elem = fm_in[tii*fsize_in + (trr)*fdim_in + (tcc)];
                                l_fmap[_trr][_tcc][_tii] = fm_elem;
                            }
                        }
                    }
                    for (int trr=row, _trr=0; trr<min(row+TR, fdim_in+offset); ++trr, ++_trr) {
                        for (int tcc=col, _tcc=0; tcc<min(col+TC, fdim_in+offset); ++tcc, ++_tcc) {
                            for (int too=outf, _too=0; too<min(outf+TOUT, conv_size_out); ++too, ++_too) {
                                l_out_fmap[_trr][_tcc][_too] = 0.f;
                            }
                        }
                    }
                    // printf("inf: %d, row=%d, col=%d\n", inf, row, col);
                    // for (int trr=0; trr<TR+ksize-1; ++trr) {
                    //     for (int tcc=0; tcc<TC+ksize-1; ++tcc) {
                    //         printf("%.2f ", l_fmap[trr][tcc][0]);
                    //     }
                    //     printf("\n");
                    // }
                    // printf("\n");
                    // Convolution
                    for (int k=0; k<ksize; ++k) {
                        for (int l=0; l<ksize; ++l) {
                            for (int trr=row, _trr=0; trr<min(row+TR, fdim_in-offset) && _trr<TR; trr+=strides, _trr+=strides) {
                                for (int tcc=col, _tcc=0; tcc<min(col+TC, fdim_in-offset) && _tcc<TC; tcc+=strides, _tcc+=strides) {
                                    for (int too=outf, _too=0; too<min(outf+TOUT, conv_size_out); ++too, ++_too) {
                                        for (int tii=inf, _tii=0; tii<min(inf+TIN, conv_size_in); ++tii, ++_tii) {
                                            l_out_fmap[_trr][_tcc][_too] += l_fmap[_trr+k][_tcc+l][_tii] * l_weights[k][l][_too][_tii];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    // if (strides==2 & inf==0) {
                    // printf("inf: %d, row=%d, col=%d\n", inf, row, col);
                    // for (int trr=0; trr<TR; ++trr) {
                    //     for (int tcc=0; tcc<TC; ++tcc) {
                    //         printf("%.2f ", l_out_fmap[trr][tcc][0]);
                    //     }
                    //     printf("\n");
                    // }
                    // printf("\n");
                    // }
                    int _offset = (strides==2)?0:offset;
                    // Store in output fmap
                    for (int trr=row, _trr=0; trr<min(row+TR, fdim_out-2*offset) && _trr<TR; ++trr, _trr+=strides) {
                        for (int tcc=col, _tcc=0; tcc<min(col+TC, fdim_out-2*offset) && _tcc<TC; ++tcc, _tcc+=strides) {
                            for (int too=outf, _too=0; too<min(outf+TOUT, conv_size_out); ++too, ++_too) {
                                    //if (inf==0) printf("%d, %d, %d\n", _trr, _tcc, _too);
                                    if(inf==0) {
                                        float fm_elem = l_out_fmap[_trr][_tcc][_too];
                                        fm_out[too*fsize_out + (trr+_offset)*fdim_out + (tcc+_offset)] = fm_elem; 
                                    } else {
                                        fm_out[too*fsize_out + (trr+_offset)*fdim_out + (tcc+_offset)] += l_out_fmap[_trr][_tcc][_too];
                                    }
                            }
                        }
                    }
                }
            }
        }
    }
}

/*
__kernel void pe_ff( const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const float* conv_kernel,
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
            fm_out[outf*fsize_out + _i/strides*fdim_out + _j/strides] = acc;
        }
    }
}
*/
/*
__kernel void pe_tile_ff( const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const float* conv_kernel,
                __global const float* fm_in, __global float* fm_out){

    // conv consts
    const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize;         // TODO: avoid multiplication in kernel
    const int offset = ksize/2;
    
    // fm consts
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);
    const int global_i = TILE_SIZE*get_group_id(0) + local_i;
    const int global_j = TILE_SIZE*get_group_id(1)+ local_j;

    __local float fm_in_local[TILE_SIZE+2][TILE_SIZE+2];
    __local float kern_in_local[3][3];
    float acc;

    const int n_tiles = conv_size_out/TILE_SIZE;
    for(int outf=0; outf<conv_size_out; ++outf){
            acc = 0.0f;
            for(int inf=0; inf<conv_size_in; ++inf){
                // Load into shared memory
                int ii = TILE_SIZE*get_group_id(0) + 2*local_i - 1;
                int jj = TILE_SIZE*get_group_id(1) + 2*local_j - 1;
                for(int li=0; li<2; ++li){
                    for(int lj=0; lj<2; ++lj){
                        if(2*local_i+li>=TILE_SIZE+2) continue;
                        if(2*local_j+lj>=TILE_SIZE+2) continue;

                        float fm_elem;
                        if((ii+li<0)||(jj+lj<0)||(ii+li>=fdim_in)||(jj+lj>=fdim_in)) fm_elem = 0.0f;
                        else fm_elem = fm_in[inf*fsize_in + (ii+li)*fdim_in + (jj+lj)];
                        fm_in_local[2*local_i+li][2*local_j+lj] = fm_elem;
                    }
                }
                if(local_i<3 && local_j<3){ //TODO: make independent from TILE_SIZE
                    int k = local_i;
                    int l = local_j;
                    kern_in_local[k][l] = conv_kernel[k*xsize + l*ysize + inf*zsize + outf];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                int i = local_i; int j = local_j;
                for(int k=0; k<3; ++k){
                    for(int l=0; l<3; ++l){
                        float fm_elem = fm_in_local[i+k][j+l];                                              
                        acc += kern_in_local[k][l] * fm_elem;
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            fm_out[outf*fsize_out + global_i*fdim_out + global_j] = acc;
        }
}*/


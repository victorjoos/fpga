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
#pragma OPENCL EXTENSION cl_intel_channels : enable

#define TILE_SIZE 1
#define KERNEL_SIZE 3
#define NUM_LANES 4
#define MASK_WEIGHT 0xffffffff
#define MASK_ACT 0xffffffff

typedef float data_t;

typedef struct {
    data_t data[KERNEL_SIZE][KERNEL_SIZE];
} vec_weight_t;

typedef struct {
    data_t data[TILE_SIZE+2];
} vec_fmap_t;

typedef struct {
    vec_weight_t lane[NUM_LANES];
} vec_w_lane_t;

typedef struct {
    vec_fmap_t lane[NUM_LANES];
} vec_fmap_lane_t;

typedef struct {
    data_t lane[NUM_LANES];
} lane_t;

#ifdef FPGA_EMU
channel vec_fmap_t fmap_channel __attribute((depth(999999999)));
channel vec_weight_t weight_channel __attribute((depth(999999999)));
// channel vec_weight_t weight_channel;
channel data_t conv_channel __attribute((depth(999999999)));
channel data_t bn_channel;
channel vec_fmap_t act_channel;
#else
channel vec_fmap_t fmap_channel __attribute((depth(3)));
channel vec_weight_t weight_channel __attribute((depth(0)));
// channel vec_weight_t weight_channel;
channel data_t conv_channel __attribute((depth(16)));
channel data_t bn_channel;
channel vec_fmap_t act_channel;
#endif




 #define ksize 3
// #define PATCH_SIZE 6
__kernel void load_mem(const int conv_size_in, const int conv_size_out,
                        const int _ksize,
                        const int fdim_in, const int fdim_out,
                        __global const float* restrict conv_kernel,
                        __global const float* restrict fm_in) {
    
    const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize;         // TODO: avoid multiplication in kernel
    const int offset = ksize/2;
    const int fsize_in = fdim_in * fdim_in;
    // printf("loading mem ?\n");
    int number_of_writes=0;
    // printf("hello from load_mem with fdim: %d; number of fmaps: %d\n", fdim_in, conv_size_out);

    __local vec_fmap_t fmap_line;
    __local volatile vec_weight_t weights; // turn to vector of NUM_LANES
    for(int outf=0; outf<conv_size_out; ++outf) {/* Loop over all output fmaps */
        for(int inf=0; inf<conv_size_in; ++inf) {
            /* Send one kernel */
            for (int k=0; k<ksize; ++k) {
                for (int l=0; l<ksize; ++l) {
                    int place = k*xsize + l*ysize + inf*conv_size_out + outf;
                    weights.data[k][l] = conv_kernel[place];
                }
            }
            write_channel_intel(weight_channel, weights);
            // printf("%d\n", number_of_writes++);
            // printf("wrote kernel to channel for in:%d, out:%d\n", inf, outf);
            for (int ii=-offset; ii<fdim_in-offset;++ii) {
                for (int jj=-offset; jj<fdim_in+offset; jj+=1/*ksize*/) {
                    for (int kk=0; kk<TILE_SIZE+2; ++kk) {
                        data_t fm_elem;
                        if ((ii+kk<0) || (jj<0) || (ii+kk>=fdim_in) || (jj>=fdim_in)) fm_elem = 0;
                        else fm_elem = fm_in[inf*fsize_in + (ii+kk)*fdim_in + (jj)];
                        fmap_line.data[kk] = fm_elem;
                    }
                    write_channel_intel(fmap_channel, fmap_line);
                    number_of_writes++;
                    // printf("%d\n", number_of_writes++);
                }
            }
        }
        // printf("[load_mem] got to outf: %d\n", outf);
    }
    // printf("[load_mem] end of load_mem %d\n", number_of_writes);
}
__kernel void pe_ff_pipe(const int conv_size_in, const int conv_size_out,
                        const int _ksize, const int strides,
                        const int fdim_in, const int fdim_out) {
    

    __local vec_weight_t fmap_tile;
    __local vec_weight_t weights;
    int number_of_reads = 0;
    int number_of_writes = 0;
    // printf("hello from pe_ff_pipe with fdim: %d; number of fmaps: %d\n", fdim_in, conv_size_out);

    for(int outf=0; outf<conv_size_out; ++outf) {/* Loop over all output fmaps */
        for(int inf=0; inf<conv_size_in; ++inf) {
            
            weights = read_channel_intel(weight_channel);
            for (int ii=0; ii<fdim_in; ++ii) {
                /* load first elements from channel */
                /*for (int kk=1; kk<ksize; ++kk) {
                    vec_fmap_t in_channel = read_channel_intel(fmap_channel);
                    number_of_reads++;
                    for (int i=0; i<TILE_SIZE+2; ++i) {
                        fmap_tile.data[i][kk] = in_channel.data[i];
                    }
                }*/
                for (int jj=-ksize+1; jj<fdim_in; ++jj) {
                    vec_fmap_t temp = read_channel_intel(fmap_channel);
                    number_of_reads++;
                    for (int i=0; i<TILE_SIZE+2; ++i) {
                        fmap_tile.data[i][0] = fmap_tile.data[i][1];
                        fmap_tile.data[i][1] = fmap_tile.data[i][2];
                        fmap_tile.data[i][2] = temp.data[i];
                    }
                    if (jj<0) continue;
                    data_t acc = 0;
                    for (int ki=0; ki<ksize; ki++) {
                        for (int kj=0; kj<ksize; kj++) {
                            acc += weights.data[ki][kj] * fmap_tile.data[ki][kj];
                        }
                    }
                    write_channel_intel(conv_channel, acc);
                    number_of_writes++;
                    // printf("%d\n", number_of_writes++);
                }
            }
        }
        // printf("[pe_ff] got to fmap_out: %d\n", outf);
    }
    // printf("[pe_ff] end of pe ff %d, %d\n", number_of_reads, number_of_writes);
}

__kernel void mem_write(const int conv_size_in, const int conv_size_out,
                        const int fdim_in, const int fdim_out,
                        __global data_t* restrict fm_out) {
    const int fsize_out = fdim_out * fdim_out;
    
    int number_of_reads = 0;
    // printf("hello from mem_write with fdim: %d; number of fmaps: %d\n", fdim_in, conv_size_out);

    for (int outf=0; outf<conv_size_out; ++outf) {
        for (int inf=0; inf<conv_size_in; ++inf) {
            for (int ii=0; ii<fdim_in; ++ii) {
                for (int jj=0; jj<fdim_in; ++jj) {
                    fm_out[outf*fsize_out + ii*fdim_out + jj] += read_channel_intel(conv_channel);
                    number_of_reads++;
                    // printf("got one whole fmap ready for out[%d][%d]=%f!\n", ii, jj, fm_out[outf*fsize_out + ii*fdim_out + jj]);
                }
            }
        }
        // printf("[mem_write] got to fmap_out: %d\n", outf);
    }
    // printf("write ended %d\n", number_of_reads);
}

__kernel void pe_ff( const int conv_size_in, const int conv_size_out,
                const int _ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const float* conv_kernel,
                __global const float* fm_in, __global float* fm_out){
    const int outf = get_global_id(0);
    // printf("hello from conv\n");
    // conv consts
    const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*_ksize;         // TODO: avoid multiplication in kernel
    const int offset = _ksize/2;
    
    // fm consts
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    
    for(int _i=(strides==2)?offset:0; _i<fdim_in; _i+=strides){
        for(int _j=(strides==2)?offset:0; _j<fdim_in; _j+=strides){
            int i = _i-offset; 
            int j = _j-offset;
            float acc = 0.0f;
            for(int inf=0; inf<conv_size_in; ++inf){
                for(int k=0; k<_ksize; ++k){
                    for(int l=0; l<_ksize; ++l){
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
}
*/

// #include "config.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable
#define TR 4 // use TR == TC ?
#define TC 4
#define TOUT 2
#define TIN  2
#define MAX_KSIZE 3
#define MAX_TOUT 128
#define MAX_TIN 128

channel uchar weights_channel;
channel short fmaps_channel;

__kernel void load_weights(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const uchar* restrict conv_kernel) {
	const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize; 
    const int offset = ksize/2;

    
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    const int max_conv_size = ksize*ksize*conv_size_in*conv_size_out;
    const bool is_strided = (strides==2);
    
    __local uchar l_weights[MAX_KSIZE][MAX_KSIZE][MAX_TOUT][MAX_TIN];
    int iter = 0;
    #pragma ivdep array(l_weights)
    for (int row=(is_strided)?0:-offset; row<fdim_in-offset; row += TR) {
        #pragma ivdep array(l_weights)
        for (int col=(is_strided)?0:-offset; col<fdim_in-offset; col += TC) {
            #pragma ivdep array(l_weights)
            for (int outf=0; outf<conv_size_out; outf += TOUT) {
                const int __too_limit = min(TOUT, conv_size_out-outf);
                #pragma ivdep array(l_weights)
                for (int inf=0; inf<conv_size_in; inf += TIN) {
                    // load memory here ...
                    // Load weights
                    const int _tii_limit = min(TIN,  conv_size_in-inf);
                    const int _too_limit = __too_limit;
                    #pragma ivdep array(l_weights)
                    for (int k=0; k<ksize; ++k) {
                        #pragma ivdep array(l_weights)
                        for (int l=0; l<ksize; ++l) {
                            #pragma ivdep array(l_weights)
                            for (int too=outf, _too=0; _too<TOUT; ++too, ++_too) {
                                #pragma ivdep array(l_weights)
                                for (int tii=inf, _tii=0; _tii<TIN; ++tii, ++_tii) {
                                    uchar weight;
                                    if (iter==0) {
                                        if(_too<_too_limit && _tii<_tii_limit) weight = conv_kernel[k*xsize + l*ysize + too*zsize + tii];
                                        else weight = 0b10;
                                        l_weights[k][l][too][tii] = weight;
                                    } else {
                                        weight = l_weights[k][l][too][tii];
                                    }
                                    write_channel_intel(weights_channel, weight);
                                }
                            }
                        }
                    }
                }
            }
            if (iter==0) mem_fence(CLK_LOCAL_MEM_FENCE);
            iter = 1;
        }
    }
}

__kernel void load_fmaps(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const short* restrict fm_in) {
    const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize; 
    const int offset = ksize/2;

    
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    const int max_conv_size = ksize*ksize*conv_size_in*conv_size_out;
    const bool is_strided = (strides==2);
    
    // __local uchar l_fmap[TIN][TR+MAX_KSIZE-1][TC+MAX_KSIZE-1];
    for (int row=(is_strided)?0:-offset; row<fdim_in-offset; row += TR) {
        for (int col=(is_strided)?0:-offset; col<fdim_in-offset; col += TC) {
            for (int outf=0; outf<conv_size_out; outf += TOUT) {
                for (int inf=0; inf<conv_size_in; inf += TIN) {
                    // load memory here ...
                    // Load weights
                    const int _tii_limit = min(TIN,  conv_size_in-inf);
                    const int _too_limit = min(TOUT, conv_size_out-outf);
                    const int _trr_limit = min(TR+ksize-1, fdim_in+offset-row);
                    const int _tcc_limit = min(TC+ksize-1, fdim_in+offset-col);
                    for (int tii=inf, _tii=0; _tii<TIN; ++tii, ++_tii) {
                        if(_tii<_tii_limit){
                            for (int trr=row, _trr=0; _trr<TR+MAX_KSIZE-1; ++trr, ++_trr) {
                                if(_trr<_trr_limit){
                                    for (int tcc=col, _tcc=0; _tcc<TC+MAX_KSIZE-1; ++tcc, ++_tcc) {
                                        if(_tcc<_tcc_limit){
                                            short fm_elem;
                                            if((trr<0)||(tcc<0)||(trr>=fdim_in)||(tcc>=fdim_in)) fm_elem = 0;
                                            else fm_elem = fm_in[tii*fsize_in + (trr)*fdim_in + (tcc)];
                                            write_channel_intel(fmaps_channel, fm_elem);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


__kernel void pe_ff(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const uchar* restrict conv_kernel,
                __global const short* restrict fm_in, __global short* restrict fm_out){
    // printf("[KERNEL] ksize: %d, strides: %d, conv_size_in: %d, conv_size_out: %d, fdim_in: %d, fdim_out: %d\n", ksize, strides, conv_size_in, conv_size_out, fdim_in, fdim_out);
    const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize; 
    const int offset = ksize/2;

    
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    const int max_conv_size = ksize*ksize*conv_size_in*conv_size_out;
    const bool is_strided = (strides==2);
    __local short l_out_fmap[TOUT][TR][TC];
    __local uchar l_weights[MAX_KSIZE][MAX_KSIZE][TOUT][TIN];
    __local short l_fmap[TIN][TR+MAX_KSIZE-1][TC+MAX_KSIZE-1];
    for (int row=(is_strided)?0:-offset; row<fdim_in-offset; row += TR) {
        for (int col=(is_strided)?0:-offset; col<fdim_in-offset; col += TC) {
            for (int outf=0; outf<conv_size_out; outf += TOUT) {

                // Load psums here
                for (int _too=0; _too<TOUT; ++_too) {
                    for (int _trr=0; _trr<TR; ++_trr) {
                        for (int _tcc=0; _tcc<TC; ++_tcc) {
                            l_out_fmap[_too][_trr][_tcc] = 0;
                        }
                    }
                }


                const int __too_limit = min(TOUT, conv_size_out-outf);
                for (int inf=0; inf<conv_size_in; inf += TIN) {
                    // load memory here ...
                    // Load weights
                    const int _tii_limit = min(TIN,  conv_size_in-inf);
                    const int _too_limit = __too_limit;
                    for (int k=0; k<ksize; ++k) {
                        for (int l=0; l<ksize; ++l) {
                            for (int tii=inf, _tii=0; _tii<TIN; ++tii, ++_tii) {
                                for (int too=outf, _too=0; _too<TOUT; ++too, ++_too) {
                                    l_weights[k][l][_too][_tii] = read_channel_intel(weights_channel);
                                }
                            }
                        }
                    }

                    // Load fmaps
                    const int _trr_limit = min(TR+ksize-1, fdim_in+offset-row);
                    const int _tcc_limit = min(TC+ksize-1, fdim_in+offset-col);
                    for (int tii=inf, _tii=0; _tii<TIN; ++tii, ++_tii) {
                        if(_tii<_tii_limit){
                            for (int trr=row, _trr=0; _trr<TR+MAX_KSIZE-1; ++trr, ++_trr) {
                                if(_trr<_trr_limit){
                                    for (int tcc=col, _tcc=0; _tcc<TC+MAX_KSIZE-1; ++tcc, ++_tcc) {
                                        /*if(_tcc<_tcc_limit){
                                            short fm_elem;
                                            if((trr<0)||(tcc<0)||(trr>=fdim_in)||(tcc>=fdim_in)) fm_elem = 0;
                                            else fm_elem = fm_in[tii*fsize_in + (trr)*fdim_in + (tcc)];
                                            l_fmap[_tii][_trr][_tcc] = fm_elem;
                                        }*/
                                        l_fmap[_tii][_trr][_tcc] = read_channel_intel(fmaps_channel);
                                    }
                                }
                            }
                        }
                    }

                    // Convolution
                    const int _tii_limit_copy = _tii_limit; //necessary???
                    for (int k=0; k<ksize; ++k) {
                        for (int l=0; l<ksize; ++l) {
                            for (int _too=0; _too<TOUT; ++_too) {
                                for (int _tii=0; _tii<TIN; ++_tii) {
                                    if(_tii<_tii_limit_copy){
                                        uchar ck_elem = l_weights[k][l][_too][_tii];
                                        if(~(ck_elem>>1)&0b1){ // weight is zero so no computation is required
                                            for (int _trr=0; _trr<TR; ++_trr) {
                                                for (int _tcc=0; _tcc<TC; ++_tcc) {
                                                    short fm_elem = l_fmap[_tii][_trr+k][_tcc+l];
                                                    if(!ck_elem) fm_elem = -fm_elem;
                                                    l_out_fmap[_too][_trr][_tcc] += fm_elem;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }


                // Store in output fmap
                const int _offset = (is_strided)?0:offset;
                const int _too_limit_copy = __too_limit;
                for (int too=outf, _too=0; _too<TOUT; ++too, ++_too) {
                    if(_too<_too_limit_copy){
                        for (int trr=row/strides, _trr=0; trr<min(row+TR, fdim_out-offset) && _trr<TR; ++trr, _trr+=strides) {
                            for (int tcc=col/strides, _tcc=0; tcc<min(col+TC, fdim_out-offset) && _tcc<TC; ++tcc, _tcc+=strides) {
                                short fm_elem = l_out_fmap[_too][_trr][_tcc];
                                fm_out[too*fsize_out + (trr+_offset)*fdim_out + (tcc+_offset)] = fm_elem; 
                            }
                        }
                    }
                }
            }
        }
    }
}




/*
__kernel void pe_ff( const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const uchar* conv_kernel,
                __global const short* fm_in, __global short* fm_out){
    const int outf = get_global_id(0);
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
            short acc = 0;
            for(int k=0; k<ksize; ++k){
                for(int l=0; l<ksize; ++l){
                    for(int inf=0; inf<conv_size_in; ++inf){
                        if((i+k<0)||(j+l<0)||(i+k>=fdim_in)||(j+l>=fdim_in)) continue; // element is out of bounds => 0
                        uchar ck_elem = conv_kernel[k*xsize + l*ysize + inf*zsize + outf];
                        if(ck_elem>>1) continue; // weight is zero so no computation is required
                        short fm_elem = fm_in[inf*fsize_in + (i+k)*fdim_in + (j+l)];
                        if (first){ // fixed point
                            if(!ck_elem) fm_elem = -fm_elem;
                            acc += fm_elem;
                        } else { // TODO: later on transform to double popcount (one for negatives, other one for positives)
                            if(!ck_elem) fm_elem = -fm_elem;
                            acc += fm_elem;
                        }
                    }
                }
            }
            fm_out[outf*fsize_out + _i/strides*fdim_out + _j/strides] = acc;
        }
    }
} */

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

// #include "config.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable
#define TR 16 // use TR == TC ?
#define TC 16
#define TOUT 32
#define TIN  32
#define MAX_KSIZE 3
#define MAX_TOUT 128
#define MAX_TIN 128

typedef struct bn_vals{
    char gamma;
    char gsign;
    short beta;
} bn_vals_t;

#ifndef FPGA_EMU
    channel uchar weights_channel;// __attribute__((depth(TIN*TOUT*MAX_KSIZE*MAX_KSIZE*2)));
    channel short fmaps_channel;// __attribute__((depth(TR*TC*TIN*2)));
    channel short out_fmaps_channel;
    channel struct bn_vals bns_channel;
#else
    channel uchar weights_channel __attribute__((depth(9999999999999)));
    channel short fmaps_channel __attribute__((depth(9999999999999)));
    channel short out_fmaps_channel __attribute__((depth(9999999999999)));
    channel struct bn_vals bns_channel __attribute__((depth(128*16*16)));
#endif

__kernel void load_weights(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const uchar* restrict conv_kernel) {
    // printf("hello from load_weights\n");
	const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize; 
    const int offset = ksize/2;

    
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    const int max_conv_size = ksize*ksize*conv_size_in*conv_size_out;
    const bool is_strided = (strides==2);
    
    // __local uchar l_weights[MAX_KSIZE][MAX_KSIZE][MAX_TOUT][MAX_TIN];
    // int iter = 0;
    // #pragma ivdep array(l_weights)
    #pragma loop_coalesce 3
    for (int row=(is_strided)?0:-offset; row<fdim_in-offset; row += TR) {
        // #pragma ivdep array(l_weights)
        for (int col=(is_strided)?0:-offset; col<fdim_in-offset; col += TC) {
            // #pragma ivdep array(l_weights)
            for (int outf=0; outf<conv_size_out; outf += TOUT) {
                const int __too_limit = min(TOUT, conv_size_out-outf);
                // #pragma ivdep array(l_weights)
                for (int inf=0; inf<conv_size_in; inf += TIN) {
                    // load memory here ...
                    // Load weights
                    const int _tii_limit = min(TIN,  conv_size_in-inf);
                    const int _too_limit = __too_limit;
                    // #pragma ivdep array(l_weights)
                    for (int k=0; k<ksize; ++k) {
                        // #pragma ivdep array(l_weights)
                        for (int l=0; l<ksize; ++l) {
                            // #pragma ivdep array(l_weights)
                            for (int tii=inf, _tii=0; _tii<TIN; ++tii, ++_tii) {
                                // #pragma ivdep array(l_weights)
                                for (int too=outf, _too=0; _too<TOUT; ++too, ++_too) {
                                    uchar weight;
                                    // if (iter==0) {
                                        if(_too<_too_limit && _tii<_tii_limit) weight = conv_kernel[k*xsize + l*ysize + tii*zsize + too];
                                        else weight = 0b10;
                                    //     l_weights[k][l][too][tii] = weight;
                                    // } else {
                                    //     weight = l_weights[k][l][too][tii];
                                    // }
                                    write_channel_intel(weights_channel, weight);
                                }
                            }
                        }
                    }
                }
            }
            // if (iter==0) mem_fence(CLK_LOCAL_MEM_FENCE);
            // iter = 1;
        }
    }
}

__kernel void load_fmaps(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const short* restrict fm_in) {
    // printf("hello from load_fmaps\n");
    const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize; 
    const int offset = ksize/2;

    
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    const int max_conv_size = ksize*ksize*conv_size_in*conv_size_out;
    const bool is_strided = (strides==2);
    
    
    #pragma loop_coalesce 4
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
    // printf("DOOOOOOOOOOOOOOOOOOOOOOOONE");
}

__kernel void load_bns(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const bn_vals_t* restrict bn_values) {

    const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize; 
    const int offset = ksize/2;
    const bool is_strided = (strides==2);
    
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    const int max_conv_size = ksize*ksize*conv_size_in*conv_size_out;
    
    #pragma loop_coalesce 3
    for (int row=(is_strided)?0:-offset; row<fdim_in-offset; row += TR) {
        for (int col=(is_strided)?0:-offset; col<fdim_in-offset; col += TC) {
            for (int outf=0; outf<conv_size_out; ++outf) {
                // const int _too_limit_copy = min(TOUT, conv_size_out-outf);
                // for (int too=outf, _too=0; _too<TOUT; ++too, ++_too) {
                // if(_too<_too_limit_copy){
                bn_vals_t bv = bn_values[outf];
                write_channel_intel(bns_channel, bv);
                // }
            }
        }
    }
}


__kernel void pe_ff(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out){
    // printf("[KERNEL] ksize: %d, strides: %d, conv_size_in: %d, conv_size_out: %d, fdim_in: %d, fdim_out: %d\n", ksize, strides, conv_size_in, conv_size_out, fdim_in, fdim_out);
    // printf("hello from pe_ff\n");
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
    #pragma loop_coalesce 3
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
                                        if(_tcc<_tcc_limit)
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
                                            // #pragma unroll
                                            for (int _trr=0; _trr<TR; ++_trr) {
                                                #pragma unroll
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
                                write_channel_intel(out_fmaps_channel, fm_elem);
                            }
                        }
                    }
                }
            }
        }
    }
}


__kernel void write_fmaps(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global short* restrict fm_out){
    // printf("hello from pe_ff\n");
    const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize; 
    const int offset = ksize/2;
    
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    const int max_conv_size = ksize*ksize*conv_size_in*conv_size_out;
    const bool is_strided = (strides==2);
    #pragma loop_coalesce 3
    for (int row=(is_strided)?0:-offset; row<fdim_in-offset; row += TR) {
        for (int col=(is_strided)?0:-offset; col<fdim_in-offset; col += TC) {
            for (int outf=0; outf<conv_size_out; outf += 1) {
                // Store in output fmap
                const int _offset = (is_strided)?0:offset;
                struct bn_vals bv = read_channel_intel(bns_channel);
                char gamma = bv.gamma;
                char gsign = bv.gsign;
                short beta = bv.beta;
                uchar abs_gamma = (gamma<0)?-gamma:gamma;
                bool is_gamma_neg = (gamma<0);
                for (int trr=row/strides, _trr=0; trr<min(row+TR, fdim_out-offset) && _trr<TR; ++trr, _trr+=strides) {
                    for (int tcc=col/strides, _tcc=0; tcc<min(col+TC, fdim_out-offset) && _tcc<TC; ++tcc, _tcc+=strides) {
                        short fm_elem = read_channel_intel(out_fmaps_channel);
                        // Apply BN
                        bool is_fe_neg = fm_elem<0;
                        ushort ufe = (is_fe_neg)?-fm_elem:fm_elem;
                        if(!first)  ufe = ufe << 8;
                        if(is_gamma_neg) ufe = ufe >> abs_gamma;
                        else             ufe = ufe << abs_gamma;
                        
                        fm_elem = ufe;
                        char fsign =(is_fe_neg)?1: 0;
                        if(fsign ^ gsign) fm_elem = -fm_elem;
                        fm_elem += beta;

                        // Activate
                        if(fm_elem<0){
                            if( fm_elem < -128) fm_elem =  0;
                            else         fm_elem = -1;
                        } else {
                            if( fm_elem > 128 ) fm_elem =  1;
                            else         fm_elem =  0;
                        }
                        fm_out[outf*fsize_out + (trr+_offset)*fdim_out + (tcc+_offset)] = fm_elem; 
                    }
                }
            }
        }
    }
}
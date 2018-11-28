// #include "config.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable
#define TR 16 // use TR == TC ?
#define TC 8
#define TOUT 32
#define TIN  16
#define MAX_KSIZE 3
#define MAX_TOUT 128
#define MAX_TIN 128

typedef struct bn_vals{
    char gamma;
    char gsign;
    short beta;
} bn_vals_t;

#ifndef FPGA_EMU
    channel uchar weights_channel       __attribute__((depth(TIN*TOUT*MAX_KSIZE*MAX_KSIZE*2)));
    channel short fmaps_channel[MAX_KSIZE*MAX_KSIZE] __attribute__((depth(TR*TC*TIN*2)));
    channel short out_fmaps_channel     __attribute__((depth(TR*TC*TOUT*2)));
    channel struct bn_vals bns_channel  __attribute__((depth(TOUT*2)));
#else
    channel uchar weights_channel __attribute__((depth(9999999999999)));
    channel short fmaps_channel[MAX_KSIZE*MAX_KSIZE] __attribute__((depth(99999999)));
    channel short out_fmaps_channel __attribute__((depth(9999999999999)));
    channel struct bn_vals bns_channel __attribute__((depth(128*16*16)));
#endif

__kernel void load_weights(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const uchar* restrict conv_kernel) {
    #ifdef FPGA_EMU
    printf("hello from load_weights\n");
    unsigned TOT = 0;
    #endif

	const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize; 
    const int offset = ksize/2;

    
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    const int max_conv_size = ksize*ksize*conv_size_in*conv_size_out;
    const bool is_strided = (strides==2);
    
    #pragma loop_coalesce 3
    for (int row=0; row<fdim_in; row += TR) {
        for (int col=0; col<fdim_in; col += TC) {
            for (int outf=0; outf<conv_size_out; outf += TOUT) {
                const int __too_limit = min(TOUT, conv_size_out-outf);
                for (int inf=0; inf<conv_size_in; inf += TIN) {
                    // load memory here ...
                    // Load weights
                    const int _tii_limit = min(TIN,  conv_size_in-inf);
                    const int _too_limit = __too_limit;

                    #pragma loop_coalesce 4
                    for (int k=0; k<ksize; ++k) {
                        for (int l=0; l<ksize; ++l) {
                            for (int tii=inf, _tii=0; _tii<_tii_limit; ++tii, ++_tii) {
                                for (int too=outf, _too=0; _too<_too_limit; ++too, ++_too) {
                                    uchar weight;
                                    /*if( && ) */ weight = conv_kernel[k*xsize + l*ysize + tii*zsize + too];
                                    // else weight = 0b10;
                                    write_channel_intel(weights_channel, weight);
                                    #ifdef FPGA_EMU
                                    TOT += 1;
                                    #endif
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    #ifdef FPGA_EMU
    printf("Pushed %d weights\n", TOT);
    #endif
}

__kernel void load_fmaps(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const short* restrict fm_in) {
    #ifdef FPGA_EMU
    printf("hello from load_fmaps\n");
    unsigned TOT[9] = {0,0,0,0,0,0,0,0,0};
    #endif

    const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize; 
    const int offset = ksize/2;

    
    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const bool is_strided = (strides==2);

    const int TR_limit = (fdim_in<TR)? fdim_in: TR;
    const int TC_limit = (fdim_in<TC)? fdim_in: TC;
    
    #pragma loop_coalesce 4
    for (int row=-offset; row<fdim_in-offset; row += TR) {
        for (int col=-offset; col<fdim_in-offset; col += TC) {
            for (int outf=0; outf<conv_size_out; outf += TOUT) {
                for (int inf=0; inf<conv_size_in; inf += TIN) {
                    // load memory here ...
                    // Load weights
                    const int _tii_limit = min(TIN,  conv_size_in-inf);
                    const int _too_limit = min(TOUT, conv_size_out-outf);
                    // const int _trr_limit = min(TR+ksize-1, fdim_in+offset-row);
                    // const int _tcc_limit = min(TC+ksize-1, fdim_in+offset-col);
                    const int _trr_limit = min(TR+ksize-1, fdim_in+offset-row);
                    const int _tcc_limit = min(TC+ksize-1, fdim_in+offset-col);
                    #pragma loop_coalesce 3
                    for (int tii=inf, _tii=0; _tii<_tii_limit; ++tii, ++_tii) {
                        for (int trr=row, _trr=0; _trr<_trr_limit; ++trr, ++_trr) {
                            for (int tcc=col, _tcc=0; _tcc<_tcc_limit; ++tcc, ++_tcc) {
                                short fm_elem;
                                if((trr<0)||(tcc<0)||(trr>=fdim_in)||(tcc>=fdim_in)) fm_elem = 0;
                                else fm_elem = fm_in[tii*fsize_in + (trr)*fdim_in + (tcc)];
                                #pragma loop_coalesce 2
                                for(unsigned short k=0; k<ksize; ++k){
                                    for(unsigned short l=0; l<ksize; ++l){
                                        int k_trr = _trr-k;
                                        int l_tcc = _tcc-l;
                                        if(k_trr>=0 && l_tcc>=0 && k_trr<TR_limit && l_tcc<TC_limit){
                                            switch(k*ksize+l){
                                                case 1: write_channel_intel(fmaps_channel[1], fm_elem); break;
                                                case 2: write_channel_intel(fmaps_channel[2], fm_elem); break;
                                                case 3: write_channel_intel(fmaps_channel[3], fm_elem); break;
                                                case 4: write_channel_intel(fmaps_channel[4], fm_elem); break;
                                                case 5: write_channel_intel(fmaps_channel[5], fm_elem); break;
                                                case 6: write_channel_intel(fmaps_channel[6], fm_elem); break;
                                                case 7: write_channel_intel(fmaps_channel[7], fm_elem); break;
                                                case 8: write_channel_intel(fmaps_channel[8], fm_elem); break;

                                                default: write_channel_intel(fmaps_channel[0], fm_elem); break;
                                            }
                                            #ifdef FPGA_EMU
                                            TOT[k*ksize + l] += 1;
                                            #endif
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
    #ifdef FPGA_EMU
    for(int i=0; i<9; ++i) printf("Put %d felems in channel %d\n", i, TOT[i]);
    #endif
}

__kernel void load_bns(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const bn_vals_t* restrict bn_values) {
    #ifdef FPGA_EMU
    unsigned TOT = 0;
    #endif

    #pragma loop_coalesce 3
    for (int row=0; row<fdim_in; row += TR) {
        for (int col=0; col<fdim_in; col += TC) {
            for (int outf=0; outf<conv_size_out; ++outf) {
                bn_vals_t bv = bn_values[outf];
                write_channel_intel(bns_channel, bv);
                #ifdef FPGA_EMU
                TOT += 1;
                #endif
            }
        }
    }
    #ifdef FPGA_EMU
    printf("%d \n", TOT);
    #endif
}


__kernel void pe_ff(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out){
    #ifdef FPGA_EMU                    
    printf("[KERNEL] ksize: %d, strides: %d, conv_size_in: %d, conv_size_out: %d, fdim_in: %d, fdim_out: %d\n", ksize, strides, conv_size_in, conv_size_out, fdim_in, fdim_out);
    printf("hello from pe_ff\n");
    unsigned TOT[9] = {0,0,0,0,0,0,0,0,0};
    unsigned TOT2 = 0;
    #endif

    const int zsize = conv_size_out;
    const int ysize = zsize*conv_size_in;
    const int xsize = ysize*ksize; 
    const int offset = ksize/2;

    const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    const int max_conv_size = ksize*ksize*conv_size_in*conv_size_out;
    const bool is_strided = (strides==2);

    #pragma loop_coalesce 3
    for (int row=-offset; row<fdim_in-offset; row += TR) {
        for (int col=-offset; col<fdim_in-offset; col += TC) {
            for (int outf=0; outf<conv_size_out; outf += TOUT) {

                short l_out_fmap[TOUT][TR][TC];

                // Load psums here
                // #pragma loop_coalesce 2
                #pragma unroll 1
                for (int _too=0; _too<TOUT; ++_too) {
                    #pragma unroll 1
                    for (int _trr=0; _trr<TR; ++_trr) {
                        #pragma unroll
                        for (int _tcc=0; _tcc<TC; ++_tcc) {
                            l_out_fmap[_too][_trr][_tcc] = 0;
                        }
                    }
                }


                const int __too_limit = min(TOUT, conv_size_out-outf);
                for (int inf=0; inf<conv_size_in; inf += TIN) {
                    // load memory here ...
                    // Load weights

                    // Convolution
                    short l_fmap[MAX_KSIZE*MAX_KSIZE][TIN][TR][TC];
                    // #pragma unroll
                    // for (ushort kl=0; kl<MAX_KSIZE*MAX_KSIZE; ++kl) {
                    //     #pragma unroll 
                    //     for (ushort _tii=0; _tii<TIN; ++_tii) {
                    //         #pragma unroll
                    //         for (int _trr=0; _trr<TR; ++_trr) {
                    //             #pragma unroll
                    //             for (int _tcc=0; _tcc<TC; ++_tcc) {
                    //                 l_fmap[kl][_tii][_trr][_tcc] = 0;
                    //             }
                    //         }
                    //     }
                    // }
                            


                    const int _too_limit = __too_limit;
                    const int _tii_limit = min(TIN,  conv_size_in-inf);
                    const int _tii_limit_copy = _tii_limit; //necessary???
                    const int _trr_limit = min(TR, fdim_in-offset-row);
                    const int _tcc_limit = min(TC, fdim_in-offset-col);
                    #pragma loop_coalesce 3
                    #pragma ivdep array(l_fmap)
                    for (ushort kl=0; kl<ksize*ksize; ++kl) {
                        #pragma ivdep array(l_fmap)
                        for (ushort _tii=0; _tii<_tii_limit_copy; ++_tii) {
                            #pragma loop_coalesce 2
                            for (int _trr=0; _trr<_trr_limit; ++_trr) {
                                for (int _tcc=0; _tcc<_tcc_limit; ++_tcc) {
                                    short value;
                                    switch(kl){
                                        case 1:  value = read_channel_intel(fmaps_channel[1]);break;
                                        case 2:  value = read_channel_intel(fmaps_channel[2]);break;
                                        case 3:  value = read_channel_intel(fmaps_channel[3]);break;
                                        case 4:  value = read_channel_intel(fmaps_channel[4]);break;
                                        case 5:  value = read_channel_intel(fmaps_channel[5]);break;
                                        case 6:  value = read_channel_intel(fmaps_channel[6]);break;
                                        case 7:  value = read_channel_intel(fmaps_channel[7]);break;
                                        case 8:  value = read_channel_intel(fmaps_channel[8]);break;
                                        default: value = read_channel_intel(fmaps_channel[0]);break;
                                    }
                                    l_fmap[kl][_tii][_trr][_tcc] = value;
                                    #ifdef FPGA_EMU
                                    TOT[kl] += 1;
                                    #endif
                                }
                            }
                            #pragma ivdep array(l_fmap)
                            for (ushort _too=0; _too<_too_limit; ++_too) {
                                // uchar ck_elem = l_weights[k][l][_too][_tii];
                                uchar ck_elem = read_channel_intel(weights_channel);
                                // mem_fence(CLK_LOCAL_MEM_FENCE);
                                if(~(ck_elem>>1)&0b1){ // weight is zero so no computation is required
                                    #pragma unroll 8
                                    for (ushort _trr=0; _trr<TR; ++_trr) {
                                        #pragma unroll
                                        for (ushort _tcc=0; _tcc<TC; ++_tcc) {
                                            short fm_elem = l_fmap[kl][_tii][_trr][_tcc];
                                            if(!ck_elem) fm_elem = -fm_elem;
                                            l_out_fmap[_too][_trr][_tcc] += fm_elem;
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
                #pragma loop_coalesce 3
                for (int too=outf, _too=0; _too<_too_limit_copy; ++too, ++_too) {
                    for (int trr=row/strides, _trr=(is_strided)?offset:0; trr<min(row+TR, fdim_out-offset) && _trr<TR; ++trr, _trr+=strides) {
                        for (int tcc=col/strides, _tcc=(is_strided)?offset:0; tcc<min(col+TC, fdim_out-offset) && _tcc<TC; ++tcc, _tcc+=strides) {
                            short fm_elem = l_out_fmap[_too][_trr][_tcc];
                            write_channel_intel(out_fmaps_channel, fm_elem);
                            #ifdef FPGA_EMU
                            TOT2 += 1;
                            #endif
                        }
                    }
                }
            }
        }
    }
    #ifdef FPGA_EMU
    for(int i=0; i<9; ++i) printf("Got %d felems from channel %d\n", i, TOT[i]);
    printf("Put %d elems in channel\n", TOT2);
    #endif
}


__kernel void write_fmaps(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global short* restrict fm_out){
    #ifdef FPGA_EMU
    printf("hello from pe_ff\n");
    unsigned TOT = 0;
    #endif

    const int offset = ksize/2;
    
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    const bool is_strided = (strides==2);
    #pragma loop_coalesce 3
    for (int row=-offset; row<fdim_in-offset; row += TR) {
        for (int col=-offset; col<fdim_in-offset; col += TC) {
            for (int outf=0; outf<conv_size_out; ++outf) {
                // Store in output fmap
                const int _offset = (is_strided)?0:offset;
                struct bn_vals bv = read_channel_intel(bns_channel);
                char gamma = bv.gamma;
                char gsign = bv.gsign;
                short beta = bv.beta;
                uchar abs_gamma = (gamma<0)?-gamma:gamma;
                bool is_gamma_neg = (gamma<0);
                #pragma loop_coalesce 2
                for (int trr=row/strides, _trr=(is_strided)?offset:0; trr<fdim_out-offset && _trr<TR; ++trr, _trr+=strides) {
                    for (int tcc=col/strides, _tcc=(is_strided)?offset:0; tcc<fdim_out-offset && _tcc<TC; ++tcc, _tcc+=strides) {
                        short fm_elem = read_channel_intel(out_fmaps_channel);
                        
                        #ifdef FPGA_EMU                        
                        TOT += 1;
                        #endif

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
                        // printf("%d, ", fm_elem);
                        if(fm_elem<0){
                            if( fm_elem > -128) fm_elem =  0;
                            else         fm_elem = -1;
                        } else {
                            if( fm_elem > 128 ) fm_elem =  1;
                            else         fm_elem =  0;
                        }
                        // printf("%d \n", fm_elem);
                        fm_out[outf*fsize_out + (trr+_offset)*fdim_out + (tcc+_offset)] = fm_elem; 
                    }
                }
            }
        }
    }
    #ifdef FPGA_EMU
    printf("%d elems in write\n", TOT);
    printf("============================\n");
    #endif
}

__kernel void add(const int size,
                    __global short* restrict fm_in1,
                    __global short* restrict fm_in2,
                    __global short* restrict fm_out){
    #pragma unroll 8
    for(int i=0; i<size; ++i){
        short fel1 = fm_in1[i];
        short fel2 = fm_in2[i];
        short fm_elem = fel1 + fel2;
        // fm_elem = fm_elem;
        if(fm_elem<0){
            if( fm_elem > -1) fm_elem =  0;
            else         fm_elem = -1;
        } else {
            if( fm_elem > 1 ) fm_elem =  1;
            else         fm_elem =  0;
        }
        fm_out[i] = fm_elem;
    }
    
}
// #include "config.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable
#define TR 8 // use TR == TC ?
#define TC 8
#define TR_MASK 0x7 // use TR == TC ?
#define TC_MASK 0x7
#define TOUT 32
#define TIN  8
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
    channel short fmaps_channel[TR][TC]         __attribute__((depth(TIN)));
    channel short out_fmaps_channel     __attribute__((depth(TR*TC*TOUT*2)));
    channel struct bn_vals bns_channel  __attribute__((depth(TOUT*2)));
#else
    channel uchar weights_channel __attribute__((depth(9999999999999)));
    channel short fmaps_channel[TR][TC] __attribute__((depth(999999999)));
    channel short out_fmaps_channel __attribute__((depth(9999999999999)));
    channel struct bn_vals bns_channel __attribute__((depth(128*16*16)));
#endif

__attribute__((max_global_work_dim(0)))
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
    

    #pragma loop_coalesce 3
    for (int row=(is_strided)?0:-offset; row<fdim_in-offset; row += TR) {
        for (int col=(is_strided)?0:-offset; col<fdim_in-offset; col += TC) {
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
                            for (int tii=inf, _tii=0; _tii<TIN; ++tii, ++_tii) {
                                for (int too=outf, _too=0; _too<TOUT; ++too, ++_too) {
                                    uchar weight;
                                    if(_too<_too_limit && _tii<_tii_limit) weight = conv_kernel[k*xsize + l*ysize + tii*zsize + too];
                                    else weight = 0b10;
                                    write_channel_intel(weights_channel, weight);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

__attribute__((max_global_work_dim(0)))
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
                    #pragma loop_coalesce 5
                    for (ushort k=0; k<ksize; ++k) {
                        for (ushort l=0; l<ksize; ++l) {
                            for (int tii=inf, _tii=0; _tii<_tii_limit; ++tii, ++_tii) {
                                for (ushort _too=0; _too<TOUT; ++_too) {
                                    short all_v[TR][TC];
                                    for (int trr=row, _trr=0; _trr<TR; ++trr, ++_trr) {
                                        for (int tcc=col, _tcc=0; _tcc<TC; ++tcc, ++_tcc) {
                                            short fm_elem;
                                            const int ktrr = trr+k;
                                            const int ltcc = tcc+l;
                                            if((ktrr<0)||(ltcc<0)||(ktrr>=fdim_in)||(ltcc>=fdim_in)) fm_elem = 0;
                                            else fm_elem = fm_in[tii*fsize_in + (trr+k)*fdim_in + (tcc+l)];
                                            all_v[_trr][_tcc] = fm_elem;
                                        }
                                    }
                                    #pragma unroll
                                    for (int _trr=0; _trr<TR; ++_trr) {
                                        #pragma unroll
                                        for (int  _tcc=0; _tcc<TC;  ++_tcc) {
                                            write_channel_intel(fmaps_channel[_trr&TR_MASK][_tcc&TC_MASK], all_v[_trr][_tcc]);
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

__attribute__((max_global_work_dim(0)))
__kernel void load_bns(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global const bn_vals_t* restrict bn_values) {
    #pragma loop_coalesce 3
    for (int row=0; row<fdim_in; row += TR) {
        for (int col=0; col<fdim_in; col += TC) {
            for (int outf=0; outf<conv_size_out; ++outf) {
                bn_vals_t bv = bn_values[outf];
                write_channel_intel(bns_channel, bv);
            }
        }
    }
}

__attribute__((max_global_work_dim(0)))
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
    __local uchar l_weights[MAX_KSIZE][MAX_KSIZE][TOUT][TIN];
    #pragma loop_coalesce 3
    for (int row=(is_strided)?0:-offset; row<fdim_in-offset; row += TR) {
        for (int col=(is_strided)?0:-offset; col<fdim_in-offset; col += TC) {
            for (int outf=0; outf<conv_size_out; outf += TOUT) {

                short l_out_fmap[TOUT][TR][TC];

                // Load psums here
                #pragma loop_coalesce 2
                for (int _too=0; _too<TOUT; ++_too) {
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
                    const int _tii_limit = min(TIN,  conv_size_in-inf);
                    const int _too_limit = __too_limit;
                    


                    // Convolution
                    const int _tii_limit_copy = _tii_limit; //necessary???
                    #pragma loop_coalesce 4
                    for (ushort k=0; k<ksize; ++k) {
                        for (ushort l=0; l<ksize; ++l) {
                            for (ushort _tii=0; _tii<_tii_limit_copy; ++_tii) {
                                for (ushort _too=0; _too<TOUT; ++_too) {
                                    uchar ck_elem = read_channel_intel(weights_channel);
                                    bool is_ck_not_zero = ~(ck_elem>>1)&0b1;
                                    bool is_ck_neg      = !ck_elem;
                                    #pragma unroll
                                    for (ushort _trr=0; _trr<TR; ++_trr) {
                                        #pragma unroll
                                        for (ushort _tcc=0; _tcc<TC; ++_tcc) {
                                            if(is_ck_not_zero){ // weight is zero so no computation is required
                                                short fm_elem = read_channel_intel(fmaps_channel[_trr][_tcc]);
                                                if(is_ck_neg) fm_elem = -fm_elem;
                                                l_out_fmap[_too][_trr][_tcc] += fm_elem;
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
                #pragma loop_coalesce 3
                for (int too=outf, _too=0; _too<_too_limit_copy; ++too, ++_too) {
                    // TODO: remove min?
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

__attribute__((max_global_work_dim(0)))
__kernel void write_fmaps(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global short* restrict fm_out){
    // printf("hello from pe_ff\n");
    // const int zsize = conv_size_out;
    // const int ysize = zsize*conv_size_in;
    // const int xsize = ysize*ksize; 
    const int offset = ksize/2;
    
    // const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    // const int max_conv_size = ksize*ksize*conv_size_in*conv_size_out;
    const bool is_strided = (strides==2);
    #pragma loop_coalesce 3
    for (int row=(is_strided)?0:-offset; row<fdim_in-offset; row += TR) {
        for (int col=(is_strided)?0:-offset; col<fdim_in-offset; col += TC) {
            for (int outf=0; outf<conv_size_out; ++outf) {
                // Store in output fmap
                const int _offset = (is_strided)?0:offset;
                struct bn_vals bv = read_channel_intel(bns_channel);
                char gamma = bv.gamma;
                char gsign = bv.gsign;
                short beta = bv.beta;
                uchar abs_gamma = (gamma<0)?-gamma:gamma;
                bool is_gamma_neg = (gamma<0);
                for (int trr=row/strides, _trr=0; trr<fdim_out-offset && _trr<TR; ++trr, _trr+=strides) {
                    for (int tcc=col/strides, _tcc=0; tcc<fdim_out-offset && _tcc<TC; ++tcc, _tcc+=strides) {
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
}

__attribute__((max_global_work_dim(0)))
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
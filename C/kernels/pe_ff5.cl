// #include "config.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable
#define TR 4
#define TC 4
#define TOUT 64
#define TIN  64

#define PR 2
#define PC 2
#define POUT 2
#define PR_MASK 0b1
#define PC_MASK 0b1
#define POUT_MASK 0b1

#define II_CYCLES 12

#define MAX_KSIZE 3
#define TOT_KSIZE 9

#define MAX_TOUT 64
#define MAX_TIN 64

typedef struct bn_vals{
    char gamma;
    char gsign;
    short beta;
} bn_vals_t;

#ifndef FPGA_EMU
    channel uchar weights_channel       __attribute__((depth(TIN*TOUT*MAX_KSIZE*MAX_KSIZE*2)));
    channel short fmaps_channel         __attribute__((depth((TR+MAX_KSIZE-1)*(TC+MAX_KSIZE-1)*TIN*2)));
    channel short out_fmaps_channel[POUT][PR][PC]     __attribute__((depth(TR*TC*TOUT*2)));
    channel struct bn_vals bns_channel  __attribute__((depth(TOUT*2)));
#else
    channel uchar weights_channel __attribute__((depth(9999999999999)));
    channel short fmaps_channel __attribute__((depth(9999999999999)));
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
                    #pragma loop_coalesce 4
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
    // const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    // const int max_conv_size = ksize*ksize*conv_size_in*conv_size_out;
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
                    for (int tii=inf, _tii=0; _tii<_tii_limit; ++tii, ++_tii) {
                        // if(_tii<_tii_limit){
                            for (int trr=row, _trr=0; _trr<_trr_limit; ++trr, ++_trr) {
                                // if(_trr<_trr_limit){
                                    for (int tcc=col, _tcc=0; _tcc<_tcc_limit; ++tcc, ++_tcc) {
                                        // if(_tcc<_tcc_limit){
                                            short fm_elem;
                                            if((trr<0)||(tcc<0)||(trr>=fdim_in)||(tcc>=fdim_in)) fm_elem = 0;
                                            else fm_elem = fm_in[tii*fsize_in + (trr)*fdim_in + (tcc)];
                                            write_channel_intel(fmaps_channel, fm_elem);
                                        // }
                                    }
                                // }
                            }
                        // }
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

    // const int zsize = conv_size_out;
    // const int ysize = zsize*conv_size_in;
    // const int xsize = ysize*ksize; 
    // const int offset = ksize/2;
    // const bool is_strided = (strides==2);
    
    // const int fsize_in = fdim_in*fdim_in; // TODO: avoid multiplication in kernel
    // const int fsize_out = fdim_out*fdim_out; // TODO: avoid multiplication in kernel
    // const int max_conv_size = ksize*ksize*conv_size_in*conv_size_out;
    
    #pragma loop_coalesce 3
    for (int row=0; row<fdim_in; row += TR) {
        for (int col=0; col<fdim_in; col += TC) {
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

__attribute__((max_global_work_dim(0)))
__kernel void pe_ff(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out){
    // printf("[KERNEL] ksize: %d, strides: %d, conv_size_in: %d, conv_size_out: %d, fdim_in: %d, fdim_out: %d\n", ksize, strides, conv_size_in, conv_size_out, fdim_in, fdim_out);
    // printf("hello from pe_ff\n");

    for (int  row=0; row<fdim_in; row+=PR) {
    #pragma unroll
    for (int _row=0;_row<PR      ;_row+=1 ) {

        for (int  col=0; col<fdim_in; col+=PC) {
        #pragma unroll
        for (int _col=0;_col<PC      ;_col+=1 ) {

            for (int  outf=0; outf<conv_size_out; outf+=POUT) {
            #pragma unroll
            for (int _outf=0;_outf<POUT         ;_outf+=1   ) {

                short shift_reg_i[II_CYCLES+1]; 
                for(int ii=0; ii<II_CYCLES+1; ++ii){
                    shift_reg_i[i] = 0;
                }
                // Calculate fmap sequentially
                #pragma unroll 1
                for (int tii=0; tii<conv_size_in; ++tii) {

                    // shift_reg_kl to store sum on kl
                    short shift_reg_kl[TOT_KSIZE+1]; 
                    for(int sri=0; sri<TOT_KSIZE+1; ++sri){
                        shift_reg_kl[i] = 0;
                    }
                    #pragma unroll 1
                    for(unsigned short kl=0; kl<TOT_KSIZE; ++kl){
                        // TODO: shared weight among kernels
                        // TODO: local shared register
                        if((ck_elem>>1)&0b1) fm_elem = 0;
                        else if(!ck_elem) fm_elem = -fm_elem;
                        shift_reg_kl[TOT_KSIZE] = /*shiftreg[0] +*/ fm_elem;
                        #pragma unroll
                        for(int srj=0; srj<TOT_KSIZE; ++srj){
                            shift_reg_kl[srj] = shift_reg_kl[srj+1];
                        }
                    }
                    short sr_acc = 0;
                    for(int sri=0; sri<TOT_KSIZE; ++sri){
                        sr_acc += shift_reg_kl[i];
                    }

                    // ==== Save new partial sum to total sum
                    shift_reg_i[II_CYCLES] = shift_reg_i[0] + sr_acc;
                    #pragma unroll
                    for(int jj=0; jj<II_CYCLES; ++jj){
                        shift_reg_kl[jj] = shift_reg_kl[jj+1];
                    }
                    
                }
                // ==== Save total sum
                short acc = 0;
                for(int jj=0; jj<II_CYCLES; ++jj){
                    acc += shift_reg_kl[jj];
                }

                // Send completed fmap to be written
                write_channel_intel(out_fmaps_channel[_row][_col][_outf], acc);

            }}
        }}
    }}
}

__attribute__((max_global_work_dim(0)))
__kernel void write_fmaps(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __global short* restrict fm_out){
    // printf("hello from pe_ff\n");

    const int offset = ksize/2; 
    const int fsize_out = fdim_out*fdim_out;
    const bool is_strided = (strides==2);
    for (int  outf=0; outf<conv_size_out; outf+=POUT) {
    for (int  row=0; row<fdim_in; row+=PR) {
    for (int  col=0; col<fdim_in; col+=PC) {
        for (int _outf=0;_outf<POUT         ;_outf+=1   ) {
            struct bn_vals bv = read_channel_intel(bns_channel);
            char gamma = bv.gamma;
            char gsign = bv.gsign;
            short beta = bv.beta;
            uchar abs_gamma = (gamma<0)?-gamma:gamma;
            bool is_gamma_neg = (gamma<0);
            for (int _row=0;_row<PR      ;_row+=1 ) {
            for (int _col=0;_col<PC      ;_col+=1 ) {
                const int oo = outf + _outf;
                const int rr = row  + _row;
                const int cc = col  + _col;

                const int _offset = (is_strided)?0:offset;
                short fm_elem = read_channel_intel(out_fmaps_channel[_outf][_row][_col]);

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
                    if( fm_elem > -128) fm_elem =  0;
                    else         fm_elem = -1;
                } else {
                    if( fm_elem > 128 ) fm_elem =  1;
                    else         fm_elem =  0;
                }

                // Store in output fmap
                fm_out[outf*fsize_out + (trr+_offset)*fdim_out + (tcc+_offset)] = fm_elem; 
            }}
        }
    }}}
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
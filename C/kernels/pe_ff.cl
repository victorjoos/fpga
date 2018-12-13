// #include "config.h"

#define TR 4 // use TR == TC ?
#define TC 4
#define TOUT 2
#define TIN  2
#define MAX_KSIZE 3

#define TILE_SIZE 4
typedef struct bn_vals{
    char gamma;
    char gsign;
    short beta;
} bn_vals_t;
__kernel void pe_tile_ff(const int first,
                const int conv_size_in, const int conv_size_out,
                const int ksize, const int strides, 
                const int fdim_in, const int fdim_out,
                __constant const uchar* restrict conv_kernel,
                __constant short* restrict fm_in, 
                __constant bn_vals_t* restrict bn_values,
                __global short* restrict fm_out){

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
    const int global_j = TILE_SIZE*get_group_id(1) + local_j;

    const int is_strided = (strides==2);
    const unsigned int gbi = ((is_strided)? global_i/2: global_i) * fdim_out;
    const unsigned int gbj =  (is_strided)? global_j/2: global_j;


    __local short fm_in_local[TILE_SIZE+MAX_KSIZE-1][TILE_SIZE+MAX_KSIZE-1];
    __local uchar kern_in_local[MAX_KSIZE][MAX_KSIZE];
    short acc;

    const int n_tiles = conv_size_out/TILE_SIZE;
    const int should_save = !is_strided || (ksize==3 && global_i%2==1 && global_j%2==1) || (ksize==1 && global_i%2==0 && global_j%2==0);
    const int ii = TILE_SIZE*get_group_id(0) + 2*local_i - offset;
    const int jj = TILE_SIZE*get_group_id(1) + 2*local_j - offset;
    for(int outf=0; outf<conv_size_out; ++outf){
        // Load bn values
        struct bn_vals bv = bn_values[outf];
        char gamma = bv.gamma;
        char gsign = bv.gsign;
        short beta = bv.beta;
        uchar abs_gamma = (gamma<0)?-gamma:gamma;
        bool is_gamma_neg = (gamma<0);

        // Set variables for inner loop
        acc = 0;
        unsigned int sum_inf_fin = 0;
        unsigned int sum_inf_zsi = 0;
        for(int inf=0; inf<conv_size_in; ++inf){
            // Load into shared memory
            for(int li=0; li<2; ++li){
                for(int lj=0; lj<2; ++lj){
                    if(2*local_i+li>=TILE_SIZE+2) continue;
                    if(2*local_j+lj>=TILE_SIZE+2) continue;

                    short fm_elem;
                    if((ii+li<0)||(jj+lj<0)||(ii+li>=fdim_in)||(jj+lj>=fdim_in)) fm_elem = 0;
                    else fm_elem = fm_in[sum_inf_fin + (ii+li)*fdim_in + (jj+lj)];
                    fm_in_local[2*local_i+li][2*local_j+lj] = fm_elem;
                }
            }
            if(local_i<MAX_KSIZE && local_j<MAX_KSIZE){ //TODO: make independent from TILE_SIZE
                int k = local_i;
                int l = local_j;
                kern_in_local[k][l] = (k<ksize && l<ksize)? conv_kernel[k*xsize + l*ysize + sum_inf_zsi + outf]:0b10;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if(should_save){
                int i = local_i; int j = local_j;
                for(int k=0; k<MAX_KSIZE; ++k){
                    for(int l=0; l<MAX_KSIZE; ++l){
                        uchar ck_elem = kern_in_local[k][l];
                        if(~(ck_elem>>1)&0b1){ // weight is zero so no computation is required
                            short fm_elem = fm_in_local[i+k][j+l];                                              
                            if(!ck_elem) fm_elem = -fm_elem;
                            acc += fm_elem;
                        }
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            sum_inf_fin += fsize_in;
            sum_inf_zsi += zsize;
        }
        if(should_save){
            // Apply BN
            short fm_elem = acc;
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
            fm_out[outf*fsize_out + gbi + gbj] = fm_elem;
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
#include "layers.h"
#include "utils.h"
#include "cl_utils.h"
#include "activations.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "../config.h"



fm_t* convolve(conv_t* conv, fm_t* fm_in, int strides, cl_kernel* kernels){
    assert(conv->size_in == fm_in->nchannels);
    fm_t* fm_out = alloc_fm(conv->size_out, fm_in->fdim/strides);

    // set kernel arguments
    cl_kernel _kernel = (strides==1 && conv->xsize==3)? kernels[1]: kernels[0];

    cl_int ret;
    ret = clSetKernelArg(_kernel, 0, sizeof(int),    (void *)&(conv->size_in));      checkError(ret, "Failed to set args");       
    ret = clSetKernelArg(_kernel, 1, sizeof(int),    (void *)&(conv->size_out));     checkError(ret, "Failed to set args");        
    ret = clSetKernelArg(_kernel, 2, sizeof(int),    (void *)&(conv->xsize));        checkError(ret, "Failed to set args");     
    ret = clSetKernelArg(_kernel, 3, sizeof(int),    (void *)&(strides));            checkError(ret, "Failed to set args"); 
    ret = clSetKernelArg(_kernel, 4, sizeof(int),    (void *)&(fm_in->fdim));        checkError(ret, "Failed to set args");     
    ret = clSetKernelArg(_kernel, 5, sizeof(int),    (void *)&(fm_out->fdim));       checkError(ret, "Failed to set args");      
    ret = clSetKernelArg(_kernel, 6, sizeof(cl_mem), (void *)&(conv->fpga_kernel));  checkError(ret, "Failed to set args");           
    ret = clSetKernelArg(_kernel, 7, sizeof(cl_mem), (void *)&(conv->fpga_bias));    checkError(ret, "Failed to set args");         
    ret = clSetKernelArg(_kernel, 8, sizeof(cl_mem), (void *)&(fm_in->fpga_values)); checkError(ret, "Failed to set args");            
    ret = clSetKernelArg(_kernel, 9, sizeof(cl_mem), (void *)&(fm_out->fpga_values));checkError(ret, "Failed to set args");             

    // Execute the OpenCL kernel
    cl_event event;
    if(_kernel == kernels[1]){
        size_t global_size[2] = {(size_t) fm_in->fdim, (size_t) fm_in->fdim/2};
        size_t local_size[2] = {(size_t) TILE_SIZE, (size_t) TILE_SIZE};
        ret = clEnqueueNDRangeKernel(space->queue, _kernel, 2, NULL,
                global_size, local_size, 0, NULL, &event);
    }else{
        size_t global_size = (size_t) conv->size_out;
        size_t local_size = (size_t) 8;
        ret = clEnqueueNDRangeKernel(space->queue, _kernel, 1, NULL,
                &global_size, &local_size, 0, NULL, &event);
    }
    checkError(ret, "Failed enqueing kernel");
    // printf("%d, %d\n", ret, CL_SUCCESS);
    // printf("kernel started\n" );
    ret = clWaitForEvents(1, &event);
    checkError(ret, "Failed waiting for events");
    // printf("kernel finished\n");

    return fm_out;
}
fm_t* fully_connect(dense_t* dense, fm_t* fm_in){
    assert(dense->size_in == fm_in->nchannels);
    fm_t* fm_out = alloc_fm(dense->size_out, 1);
    for(int outf=0; outf<dense->size_out; ++outf){
        for(int i=0; i<fm_in->fdim; ++i){
            for(int j=0; j<fm_in->fdim; ++j){
                float acc = 0;
                for(int inf=0; inf<dense->size_in; ++inf){
                    acc += get_dense_elem(dense, inf, outf)*
                        get_fm_elem(fm_in, inf, i, j);
                }
                acc += dense->bias[outf];
                set_fm_elem(fm_out, acc, outf, i, j);
            }
        }
    }
    return fm_out;
}
fm_t* avg_pool(fm_t* fm_in){
    fm_t* fm_out = alloc_fm(fm_in->nchannels, 1);
    float* channel = fm_in->values;
    for(int n=0; n<fm_in->nchannels; ++n){
        float acc = 0.0f;
        for(int i=0; i<fm_in->fsize; ++i){
            acc += *channel;
            ++channel;
        }
        set_fm_elem(fm_out, acc/(float)fm_in->fsize, n, 0, 0);
    }
    return fm_out;
}

fm_t* apply_f(fm_t* fm_in, float(*f)(float)){
    const int size = fm_in->nchannels*fm_in->fsize;
    for(int i=0; i<size; ++i)
        fm_in->values[i] = f(fm_in->values[i]);
    return fm_in;
}
fm_t* activate(fm_t* fm_in, activation_t activ){
    float (*f)(float);
    switch(activ){
        case RELU: f = act_relu; break;
        case LEAKYRELU: f = leaky_relu; break;
        case TANH: f = act_tanh; break;
        default:  f = act_relu;
    }
    return apply_f(fm_in, f);
}
float __div(float x) {return x*0.5f;}
fm_t* divide(fm_t* fm_in){
    return apply_f(fm_in, __div);
}
fm_t* add(fm_t* fm_in1, fm_t* fm_in2){
    assert(fm_in1->nchannels == fm_in2->nchannels);
    assert(fm_in1->fdim == fm_in2->fdim);
    const int size = fm_in1->nchannels*fm_in1->fsize;
    for(int i=0; i<size; ++i)
        fm_in1->values[i] += fm_in2->values[i];
    return fm_in1;
}
fm_t* normalize(bn_t* bn, fm_t* fm_in){
    assert(bn->size == fm_in->nchannels);
    float* values = fm_in->values;
    for(int n=0; n<fm_in->nchannels; ++n){
        float beta = bn->beta[n];
        float gamma = bn->gamma[n];
        float mean = bn->mean[n];
        for(int i=0; i<fm_in->fsize; ++i){
            float x = *values;
            *values = (x-mean) * gamma + beta;
            ++values;
        }
    }
    return fm_in;
}
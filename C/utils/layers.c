#include "layers.h"
#include "utils.h"
#include "cl_utils.h"
#include "activations.h"
#include "cl_space.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "../config.h"



fm_t* convolve(conv_t* conv, fm_t* fm_in, int strides, int first, cl_kernel* kernels){
    assert(conv->size_in == fm_in->nchannels);
    fm_t* fm_out = alloc_fm(conv->size_out, fm_in->fdim/strides);

    // set kernel arguments
    cl_kernel _kernel = (strides==1 && conv->xsize==3)? kernels[1]: kernels[1];

    cl_int ret; int _karg = 0;
    ret = clSetKernelArg(_kernel, _karg++, sizeof(int),    (void *)&(first));      checkError(ret, "Failed to set args");
    ret = clSetKernelArg(_kernel, _karg++, sizeof(int),    (void *)&(conv->size_in));      checkError(ret, "Failed to set args");
    ret = clSetKernelArg(_kernel, _karg++, sizeof(int),    (void *)&(conv->size_out));     checkError(ret, "Failed to set args");
    ret = clSetKernelArg(_kernel, _karg++, sizeof(int),    (void *)&(conv->xsize));        checkError(ret, "Failed to set args");
    ret = clSetKernelArg(_kernel, _karg++, sizeof(int),    (void *)&(strides));            checkError(ret, "Failed to set args");
    ret = clSetKernelArg(_kernel, _karg++, sizeof(int),    (void *)&(fm_in->fdim));        checkError(ret, "Failed to set args");
    ret = clSetKernelArg(_kernel, _karg++, sizeof(int),    (void *)&(fm_out->fdim));       checkError(ret, "Failed to set args");
    ret = clSetKernelArg(_kernel, _karg++, sizeof(cl_mem), (void *)&(conv->fpga_kernel));  checkError(ret, "Failed to set args");
    ret = clSetKernelArg(_kernel, _karg++, sizeof(cl_mem), (void *)&(fm_in->fpga_values)); checkError(ret, "Failed to set args");
    ret = clSetKernelArg(_kernel, _karg++, sizeof(cl_mem), (void *)&(fm_out->fpga_values));checkError(ret, "Failed to set args");

    // Execute the OpenCL kernel
    cl_event event;
    if(_kernel == kernels[1]){
        size_t global_size[2] = {(size_t) fm_in->fdim, (size_t) fm_in->fdim};
        size_t local_size[2] = {(size_t) TILE_SIZE, (size_t) TILE_SIZE};
        ret = clEnqueueNDRangeKernel(space->queue[0], _kernel, 2, NULL,
                global_size, local_size, 0, NULL, &event);
    }else{
        size_t global_size = (size_t) conv->size_out;
        size_t local_size = (size_t) 8;
        ret = clEnqueueTask(space->queue[0], _kernel, 0, NULL, &event);
        // ret = clEnqueueNDRangeKernel(space->queue, _kernel, 1, NULL,
        //         &global_size, &local_size, 0, NULL, &event);
    }
    checkError(ret, "Failed enqueing kernel");
    // printf("%d, %d\n", ret, CL_SUCCESS);
    // printf("kernel started\n" );
    ret = clWaitForEvents(1, &event);
    checkError(ret, "Failed waiting for events");
    clReleaseEvent(event);
    // printf("kernel finished\n");

    return fm_out;
}
fm_t* fully_connect(dense_t* dense, fm_t* fm_in){
    assert(dense->size_in == fm_in->nchannels);
    fm_t* fm_out = alloc_fm(dense->size_out, 1);
    for(int outf=0; outf<dense->size_out; ++outf){
        for(int i=0; i<fm_in->fdim; ++i){
            for(int j=0; j<fm_in->fdim; ++j){
                cl_short acc = 0;
                for(int inf=0; inf<dense->size_in; ++inf){
                    // Not possible to xnor here due to average pooling
                    cl_uchar  dw = get_dense_elem(dense, inf, outf);
                    if(!(dw >> 1)){
                        cl_short fv = get_fm_elem(fm_in, inf, i, j);
                        if(dw) acc += fv;
                        else acc -= fv;
                    }
                }
                set_fm_elem(fm_out, acc, outf, i, j);
            }
        }
    }
    return fm_out;
}
fm_t* avg_pool(fm_t* fm_in){
    fm_t* fm_out = alloc_fm(fm_in->nchannels, 1);
    cl_short* channel = fm_in->values;
    for(int n=0; n<fm_in->nchannels; ++n){
        cl_short acc = 0;
        for(int i=0; i<fm_in->fsize; ++i){
            acc += *channel;
            ++channel;
        }
        set_fm_elem(fm_out, acc, n, 0, 0);
    }
    return fm_out;
}

fm_t* apply_f(fm_t* fm_in, cl_short(*f)(cl_short)){
    const int size = fm_in->nchannels*fm_in->fsize;
    for(int i=0; i<size; ++i)
        fm_in->values[i] = f(fm_in->values[i]);
    return fm_in;
}
fm_t* activate(fm_t* fm_in, activation_t activ){
    cl_short (*f)(cl_short);
    switch(activ){
        // case RELU: f = act_relu; break;
        // case LEAKYRELU: f = leaky_relu; break;
        // case TANH: f = act_tanh; break;
        // case BINARY: f = bin_htanh; break;
        case TERNARY: f = ter_htanh; break;
        default:  f = ter_htanh;
    }
    return apply_f(fm_in, f);
}
cl_short __div(cl_short x) {
    // x = x << 8;
    return x<<7;
}
fm_t* divide(fm_t* fm_in){
    return apply_f(fm_in, __div);
}
fm_t* add(fm_t* fm_in1, fm_t* fm_in2, cl_kernel kernel){
    assert(fm_in1->nchannels == fm_in2->nchannels);
    assert(fm_in1->fdim == fm_in2->fdim);
    
    fm_t * fm_out = alloc_fm(fm_in1->nchannels, fm_in1->fdim);
    const int size = fm_in1->fsize*fm_in1->nchannels;
    cl_int ret; cl_event event;
    int _karg = 0;
    ret = clSetKernelArg(kernel, _karg++, sizeof(int),    (void *)&(size));      checkError(ret, "Failed to set args");
    ret = clSetKernelArg(kernel, _karg++, sizeof(cl_mem), (void *)&(fm_in1->fpga_values));  checkError(ret, "Failed to set args");
    ret = clSetKernelArg(kernel, _karg++, sizeof(cl_mem), (void *)&(fm_in2->fpga_values)); checkError(ret, "Failed to set args");
    ret = clSetKernelArg(kernel, _karg++, sizeof(cl_mem), (void *)&(fm_out->fpga_values)); checkError(ret, "Failed to set args");
    ret = clEnqueueTask(space->queue[1], kernel, 0, NULL, &event);
    checkError(ret, "Failed enqueing kernel");
    ret = clWaitForEvents(1, &event);
    checkError(ret, "Failed waiting for events");
    clReleaseEvent(event);
    return fm_out;
}

fm_t* normalize(bn_t* bn, fm_t* fm_in, int first){
    assert(bn->size == fm_in->nchannels);
    cl_short* values = fm_in->values;
    for(int n=0; n<fm_in->nchannels; ++n){
        cl_short beta = bn->beta[n];
        cl_char gamma = bn->gamma[n];
        cl_char gsign = bn->gamma_sign[n];
        
        for(int i=0; i<fm_in->fsize; ++i){
            cl_short x = *values;
            cl_ushort ux = abs(x);
            cl_char xsign =(x<0)?1: 0;
            if(!first)  ux = ux << 8;
            if(gamma<0) ux = ux >> (cl_uchar)(-gamma);
            else        ux = ux << (cl_uchar)gamma;

            
            *values = ux;
            if(xsign ^ gsign) *values = -*values;
            *values += beta;
            
            ++values;
        }
    }
    return fm_in;
}
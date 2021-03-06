import sys
import os
import h5py 
import struct
import numpy as np
import re
class Bn():
    def __init__(self, data):
        self.beta = np.array(data["beta:0"]).astype('float32')
        self.gamma = np.array(data["gamma:0"]).astype('float32')
        self.mean = np.array(data["moving_mean:0"]).astype('float32')
        self.var = np.array(data["moving_variance:0"]).astype('float32')
        print(self.beta.shape[0])
        self.sizes = np.array([self.beta.shape[0]]).astype(np.int32).tobytes()
        self.all = [self.beta, self.mean, self.gamma/np.sqrt(self.var + 10**-3)]
        # self.all = [self.beta, self.gamma, self.mean, self.var]
class Conv():
    def __init__(self, data):
        self.kernel = np.array(data["kernel:0"])
        # self.bias = data["bias:0"]
        print(self.kernel.shape)
        # print(self.kernel[1][2][2][5])
        # print(self.bias[0])
        self.strides = self.kernel.shape[0]
        self.size_in = self.kernel.shape[2]
        self.size_out = self.kernel.shape[3]
        self.sizes = np.array([self.strides, self.size_in, self.size_out]).astype(np.int32).tobytes()
        self.kernel.resize(self.strides*self.strides*self.size_in*self.size_out)
        # print(self.kernel[self.strides*self.size_in*self.size_out + 2*self.size_in*self.size_out + 2*self.size_out + 5])
        self.all = [self.kernel]#, self.bias]
class Dense():
    def __init__(self, data):
        self.kernel = np.array(data["kernel:0"])
        print(self.kernel.shape)
        # self.bias = data["bias:0"]
        self.size_in = self.kernel.shape[0]
        self.size_out = self.kernel.shape[1]
        self.sizes = np.array([self.size_in, self.size_out]).astype(np.int32).tobytes()
        self.kernel.resize(self.size_in*self.size_out)
        self.all = [self.kernel]#, self.bias]

# Converts a float hdf5_dataset to a bytes array
def fldata_to_barr(fl):
    flnp = np.array(fl)
    flnp = flnp.astype('float32')
    return flnp.tobytes()#bytes(struct.pack('f'*len(fl), fl))

def write_bn(d, counter, data):
    with open(os.path.join(d, f"bn_{counter}.bin"), "wb") as fp:
        bn = Bn(data)
        fp.write(bn.sizes)
        for arr in bn.all:
            fp.write(fldata_to_barr(arr))

def write_conv(d, counter, data):
    with open(os.path.join(d, f"conv_{counter}.bin"), "wb") as fp:
        conv = Conv(data)
        # print(conv.kernel.shape)
        # print(conv.bias.shape)
        fp.write(conv.sizes)
        for arr in conv.all:
            fp.write(fldata_to_barr(arr))

def write_dense(d, counter, data):
    with open(os.path.join(d, f"dense_{counter}.bin"), "wb") as fp:
        dense = Dense(data)
        # print(dense.kernel.shape)
        # print(dense.bias.shape)
        fp.write(dense.sizes)
        for arr in dense.all:
            fp.write(fldata_to_barr(arr))


def main(file):
    d = "test2/"
    if not os.path.exists(d):
        os.makedirs(d)
    params = h5py.File(file, 'r')
    params = params["model_weights"]


    for x in params:
        group = params[x]
        print("-----")
        print(x)
        for y in group:
            print(y)
            if "batch_normalization" in y:
                match = re.match(r'(?:.*)batch_normalization_(\d*)', y)
                write_bn(d, int(match.groups()[0]), group[y])
            elif "conv2d" in y:
                match = re.match(r'(?:.*)conv2d_(\d*)', y)
                write_conv(d, int(match.groups()[0]), group[y])
            elif "dense" in y:
                match = re.match(r'(?:.*)dense_(\d*)', y)
                # write_dense(d, int(match.groups()[0]), group[y])
                write_dense(d, 1, group[y])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Need at least 1 file to parse")
        exit(-1)
    main(sys.argv[1])
    
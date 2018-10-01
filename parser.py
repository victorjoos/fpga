import sys
import os
import h5py 
import struct
import numpy as np
class Bn():
    def __init__(self, data):
        self.beta = data["beta:0"]
        self.gamma = data["gamma:0"]
        self.mean = data["moving_mean:0"]
        self.var = data["moving_variance:0"]
        self.size = self.beta.shape[0]
        self.all = [self.beta, self.gamma, self.mean, self.var]
class Conv():
    def __init__(self, data):
        self.kernel = np.array(data["kernel:0"])
        self.bias = data["bias:0"]
        self.strides = self.kernel.shape[0]
        self.size_in = self.kernel.shape[2]
        self.size_out = self.kernel.shape[3]
        self.sizes = np.array([self.strides, self.size_in, self.size_out]).tobytes()
        self.kernel.resize(self.strides*self.strides*self.size_in*self.size_out)
        self.all = [self.kernel, self.bias]
class Dense():
    def __init__(self, data):
        self.kernel = np.array(data["kernel:0"])
        self.bias = data["bias:0"]
        self.size_in = self.kernel.shape[0]
        self.size_out = self.kernel.shape[1]
        self.sizes = np.array([self.size_in, self.size_out]).tobytes()
        self.kernel.resize(self.size_in*self.size_out)
        self.all = [self.kernel, self.bias]

# Converts a float hdf5_dataset to a bytes array
def fldata_to_barr(fl):
    return np.array(fl).tobytes()#bytes(struct.pack('f'*len(fl), fl))

def write_bn(d, counter, data):
    with open(os.path.join(d, f"bn_{counter}.bin"), "wb") as fp:
        bn = Bn(data)
        fp.write(bytes(bn.size))
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
    d = "test/"
    if not os.path.exists(d):
        os.makedirs(d)
    params = h5py.File(file, 'r')
    params = params["model_weights"]

    bn_counter = 1
    conv_counter = 1
    dense_counter = 1
    for x in params:
        group = params[x]
        print("-----")
        print(x)
        for y in group:
            print(y)
            if "batch_normalization" in y:
                write_bn(d, bn_counter, group[y])
                bn_counter += 1
            elif "conv2d" in y:
                write_conv(d, conv_counter, group[y])
                conv_counter += 1
            elif "dense" in y:
                write_dense(d, dense_counter, group[y])
                dense_counter += 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Need at least 1 file to parse")
        exit(-1)
    main(sys.argv[1])
    
CC=gcc
LIB= -lm -l OpenCL# lhdf5_serial
CFLAGS= -std=c99 -g -I. #/usr/include/hdf5/serial 
MAINOBJ=main.o
OBJ=utils/utils.o utils/layers.o utils/activations.o utils/resnet.o utils/cl_utils.o
TESTOBJ=test.o tests/layers.o
NAME=main

all: $(NAME)

main: $(MAINOBJ) $(OBJ)
	$(CC) -o $@ $(MAINOBJ) $(OBJ) $(CFLAGS) $(LIB)
test: $(OBJ) $(TESTOBJ)
	$(CC) -o $@ $(TESTOBJ) $(OBJ) $(CFLAGS) -l $(LIB)
%.o: %.c #$(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
clean:
	rm $(OBJ) $(NAME)
.PHONY: clean test
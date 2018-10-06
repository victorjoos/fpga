CC=gcc
LIB= m# lhdf5_serial
CFLAGS= -std=c99 -g -Wall -I . #/usr/include/hdf5/serial 
MAINOBJ=main.o
BIN_OBJ=binary/utils.o binary/layers.o binary/activations.o binary/resnet.o
OBJ=utils/utils.o utils/layers.o utils/activations.o utils/resnet.o
TESTOBJ=test.o tests/layers.o
NAME=binary_resnet

all: $(NAME)

main: $(MAINOBJ) $(OBJ)
	$(CC) -o $@ $(MAINOBJ) $(OBJ) $(CFLAGS) -l $(LIB)
binary_resnet: $(MAINOBJ) $(BIN_OBJ)
	$(CC) -o $@ $(MAINOBJ) $(BIN_OBJ) $(CFLAGS) -l $(LIB)
test: $(OBJ) $(TESTOBJ)
	$(CC) -o $@ $(TESTOBJ) $(OBJ) $(CFLAGS) -l $(LIB)
%.o: %.c #$(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
clean:
	rm $(OBJ) $(BIN_OBJ) $(NAME)
.PHONY: clean test
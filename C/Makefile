CC=gcc
LIB= m# lhdf5_serial
CFLAGS= -std=c99 -g -Wall -I . #/usr/include/hdf5/serial 
OBJ=main.o utils/utils.o utils/layers.o utils/activations.o
NAME=main

all: $(NAME)

main: $(OBJ)
	$(CC) -o $@ $(OBJ) $(CFLAGS) -l $(LIB)
%.o: %.c #$(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
clean:
	rm $(OBJ) $(NAME)
.PHONY: clean test
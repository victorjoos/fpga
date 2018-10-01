CC=gcc
LIB=lhdf5_serial
CFLAGS=-I /usr/include/hdf5/serial -std=c99 -g
OBJ=main.o
NAME=cnn.out

all: $(NAME)

matmul: $(OBJ)
	$(CC) -o $@ $(OBJ) $(CFLAGS) -l $(LIB)
%.o: %.c #$(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
clean:
	rm $(OBJ) $(NAME)
.PHONY: clean test
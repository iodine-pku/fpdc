CC = gcc
OBJ = *.o
EXE = spmv_parallel
FLAGS = -fopenmp -lm -O3 -std=c99 -Ofast -march=native

all:${EXE}

spmv_parallel: spmv_parallel.c
	$(CC) -o $@ $^ $(FLAGS) 
clean:
	rm -f $(OBJ) $(EXE)
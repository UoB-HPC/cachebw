COMPILER=NVCC
ARCH=sm_60
SHMEM=0

COMPILER_GNU=gcc
COMPILER_INTEL=icc
COMPILER_NVCC=nvcc
CC=$(COMPILER_$(COMPILER))

CFLAGS_INTEL=-std=c11 -x$(ARCH) -qopt-zmm-usage=high -qopenmp
CFLAGS_GNU=-std=c11 -march=$(ARCH) -fopenmp
CFLAGS_NVCC=-arch=$(ARCH) -DGPU -x cu -DSHMEM=$(SHMEM)

CFLAGS = -O3 $(CFLAGS_$(COMPILER))

HEADERS = $(wildcard *.h)

default: triad

triad: cachebw.c $(HEADERS)
	$(CC) $(CFLAGS) cachebw.c -o cachebw

clean:
	-rm -rf cachebw

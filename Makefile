COMPILER=INTEL
MODEL=USM
ARCH=sapphirerapids
SHMEM=0

COMPILER_GNU=gcc
COMPILER_NVCC=nvcc

CFLAGS_GNU=-std=c11 -march=$(ARCH) -fopenmp
CFLAGS_NVCC=-arch=$(ARCH) -DGPU -x cu -DSHMEM=$(SHMEM)

ifeq (USM,$(MODEL))
	COMPILER_INTEL=icpx
	CFLAGS_INTEL=-fsycl -DSYCL_USM -DSHMEM=$(SHMEM)
else
ifeq (ACC,$(MODEL))
	COMPILER_INTEL=icpx
	CFLAGS_INTEL=-fsycl -DSYCL_ACC -DSHMEM=$(SHMEM)
else
	COMPILER_INTEL=icx
	CFLAGS_INTEL=-std=c99 -x$(ARCH) -qopt-zmm-usage=high -qopenmp
endif
endif

CC=$(COMPILER_$(COMPILER))
CFLAGS = -O3 $(CFLAGS_$(COMPILER))

HEADERS = $(wildcard *.h)

default: triad

triad: cachebw.c $(HEADERS)
	$(CC) $(CFLAGS) cachebw.c -o cachebw

clean:
	-rm -rf cachebw

MODEL=USM
OPTS=-Xlinker -z,noexecstack

COMPILER_GNU=gcc
COMPILER_NVIDIA=nvc++
COMPILER_AMD=hipcc

CFLAGS_GNU=-std=c11 -march=$(ARCH) -fopenmp
CFLAGS_NVIDIA=-gpu=$(ARCH) $(OPTS) -DCUDA -cuda -DSHMEM=$(SHMEM)
CFLAGS_AMD=-DHIP $(HIPOPTS) -DSHMEM=$(SHMEM)

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

triad: cachebw.cpp $(HEADERS)
	$(CC) $(CFLAGS) cachebw.cpp -o cachebw

clean:
	-rm -rf cachebw

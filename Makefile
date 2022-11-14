COMPILER=GNU
ARCH=armv8.4-a
SHMEM=0

COMPILER_GNU=aarch64-linux-gnu-gcc
COMPILER_INTEL=icc
COMPILER_NVCC=nvcc
CC=$(COMPILER_$(COMPILER))

CFLAGS_INTEL=-std=c99 -x$(ARCH) -qopt-zmm-usage=high -qopenmp
CFLAGS_GNU=-std=c11 -march=$(ARCH) -fopenmp
CFLAGS_NVCC=-arch=$(ARCH) -DGPU -x cu -DSHMEM=$(SHMEM)

GNU_STATIC_FLAGS=-std=c11 -O3 -fPIC -DPMU -I/home/br-rmuneeb/pmu/include -I/opt/gcc/9.3.0/snos/lib/gcc/aarch64-unknown-linux-gnu/9.3.0/include -L/home/br-rmuneeb/pmu/install/lib/ -static-libgcc -static-libstdc++
GNU_STATIC_FLAGS_NORMAL=-std=c11 -O3 -fPIC -I/home/br-rmuneeb/pmu/include -I/opt/gcc/9.3.0/snos/lib/gcc/aarch64-unknown-linux-gnu/9.3.0/include -static -static-libgcc -static-libstdc++
STATIC_OMP=/opt/gcc/9.3.0/snos/lib64/libgomp.a
CFLAGS = -O3 $(CFLAGS_$(COMPILER))

HEADERS = $(wildcard *.h)

all: cbw_static_pmu cbw_static_sve_pmu cbw_static cbw_static_sve

default: cbw_static

cbw_static_pmu: cachebw.c $(HEADERS)
	gcc $(GNU_STATIC_FLAGS) -march=armv8.4-a cachebw.c $(STATIC_OMP) -o cachebw_static_pmu -lpmu

cbw_static_sve_pmu: cachebw.c $(HEADERS)
	gcc $(GNU_STATIC_FLAGS) -march=armv8.4-a+sve cachebw.c $(STATIC_OMP) -o cachebw_static_sve_pmu -lpmu

cbw_static: cachebw.c $(HEADERS)
	gcc $(GNU_STATIC_FLAGS_NORMAL) -march=armv8.4-a cachebw.c $(STATIC_OMP) -o cachebw_static

cbw_static_sve: cachebw.c $(HEADERS)
	gcc $(GNU_STATIC_FLAGS_NORMAL) -march=armv8.4-a cachebw.c $(STATIC_OMP) -o cachebw_static_sve

clean:
	-rm -rf cachebw


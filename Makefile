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

GNU_STATIC_FLAGS=-std=c11 -O3 -fPIC -I/usr/lib/gcc-cross/aarch64-linux-gnu/11/include -static -static-libgcc -static-libstdc++
STATIC_OMP=/usr/lib/gcc-cross/aarch64-linux-gnu/11/libgomp.a
CFLAGS = -O3 $(CFLAGS_$(COMPILER))

HEADERS = $(wildcard *.h)

default: triad

triad: cachebw.c $(HEADERS)
	$(CC) $(CFLAGS) cachebw.c -o cachebw

triad_static_simeng: cachebw.c $(HEADERS)
	aarch64-linux-gnu-gcc $(GNU_STATIC_FLAGS) -march=armv8.4-a cachebw.c $(STATIC_OMP) -o cachebw_static_pmu

triad_static_sve_simeng: cachebw.c $(HEADERS)
	aarch64-linux-gnu-gcc $(GNU_STATIC_FLAGS) -march=armv8.4-a+sve cachebw.c $(STATIC_OMP) -o cachebw_static_sve

triad_x86_pmu: cachebw.c $(HEADERS)
	gcc -std=c11 -O3 -fPIC -I/usr/lib/gcc/x86_64-linux-gnu/11/include -static -static-libgcc -static-libstdc++ cachebw.c /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.a -o cachebw_static_pmu

clean:
	-rm -rf cachebw
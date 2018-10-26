CacheBW
=======

Author: Patrick Atkinson

This benchmark is primarily designed to measure the memory-bandwidth of different levels of the memory hierarchy in CPU and GPU architectures. It works by running a STREAM Triad kernel many times on each thread (CPU) or each SM (GPU), and calculates the aggregate memory-bandwidth. This methodology was inspired by the work done in:

> Deakin T, Price J, McIntosh-Smith S. Portable methods for measuring cache hierarchy performance. 2017. Poster sessions presented at IEEE/ACM SuperComputing, Denver, United States

This benchmark was first used in: 

> Matthew Martineau, Patrick Atkinson and Simon McIntosh-Smith. Benchmarking the NVIDIA V100 GPU and Tensor Cores, HeteroPar 2018.

Building
--------

CacheBW uses a single Makefile where the user will need to change a number of variables:

| Variable | Options | Default | Description                                                                                                     |
-----------|-----------------|---------------|-----------------------------------------------------------------------------------------------------------------|
| COMPILER | INTEL/GNU/NVCC  | NVCC          | Which compiler to use. GNU/INTEL for CPU, NVCC for GPU                                                          |
| ARCH     | -               | sm_60         | Which architecture to compile for. Gets passed directly to compiler.                                            |
| SHMEM    | 0/1             | 0             | Used to benchmark shared-memory on GPUs. 0 = off, 1 = on                                                        |

Running
-------

The executable is run as follows:

```
./cachebw <memory size in KiB> <no. of triad repetitions.>
```

The first argument is the size (in KiB) of the data that will be read/written per thread/SM. This is used to control which level of memory hierarchy to be examined. 

The second argument is the number of times the triad kernel is repeated, ensuring accurate results. 

**Running the benchmark a single time is rarely useful.** It is generally more interesting to run it with increasingly large data sizes, so one can view the effect of the data slipping out of each level of cache. An example of this is provided in the script 'test.sh', i.e.:

```
./test.sh
```

will print out the memory-bandwidths of each of the levels of the memory-hierarchy.

Samples Results
---------------

Sample results are provided in the results directory:

| File        | Hardware                                 |
|-------------|------------------------------------------|
|skylake.txt  | 2x Intel Platinum 8176 (2x28 cores)      |
|broadwell.txt| 2x Intel Xeon E5-2699v4 (2x22 cores)     |
|volta.txt    | NVIDIA Tesla V100 16GB                   |
|pascal.txt   | NVIDIA Tesla P100 16GB                   |

Notes
----------------

- The code can be modified to print out the memory-bandwidth per SM or thread, instead of aggregate memory-bandwidth. This may be useful data.
- Adjusting the number of thread-blocks and threads per block gives interesting results...




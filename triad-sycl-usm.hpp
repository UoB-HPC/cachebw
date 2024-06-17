#pragma once

#include <sycl/sycl.hpp>

#ifdef __SYCL_DEVICE_ONLY__
extern SYCL_EXTERNAL ulong __attribute__((overloadable))
intel_get_cycle_counter(void);
#endif

double cache_triad(size_t n, size_t nreps) {
  double tot_mem_bw = 0.0;

  try {
    sycl::queue gpuQueue{sycl::gpu_selector_v};
    sycl::device device = gpuQueue.get_device();

    // Print device
    // std::cout << "Running on " << device.get_info<sycl::info::device::name>()
    //           << "\n";

    int num_blocks =
        device.get_info<sycl::info::device::max_compute_units>();
    int block_size = 1024;

    // if we don't have enough GPU memory, don't run
    long long int max_mem =
        device.get_info<sycl::info::device::global_mem_size>();

    if (num_blocks * n * 3 * sizeof(double) > max_mem) {
      fprintf(stderr, "error: out of GPU global memory\n");
      return -1.0;
    }

    double *d_a =
        sycl::malloc_device<double>(n * num_blocks, gpuQueue);
    double *d_b =
        sycl::malloc_device<double>(n * num_blocks, gpuQueue);
    double *d_c =
        sycl::malloc_device<double>(n * num_blocks, gpuQueue);

    double *h_bw = (double *)malloc(sizeof(double) * num_blocks);
    double *d_bw =
        sycl::malloc_device<double>(num_blocks, gpuQueue);

    gpuQueue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(num_blocks * block_size, block_size), [=](sycl::nd_item<1> item) {
            const int thread_idx = item.get_local_id(0);
            const int block_idx = item.get_group(0);
            const int block_dimx = item.get_local_range(0);
            const double scalar = 2.0;

            double *a = d_a + block_idx * n;
            double *b = d_b + block_idx * n;
            double *c = d_c + block_idx * n;

            // This should place a,b,c in cache
            for (int i = thread_idx; i < n; i += block_dimx) {
              a[i] = 0.0;
              b[i] = 3.0;
              c[i] = 2.0;
            }

            ulong c0, c1;
#ifdef __SYCL_DEVICE_ONLY__
            c0 = intel_get_cycle_counter();
#endif

            for (int t = 0; t < nreps; ++t) {
              for (int i = thread_idx; i < n; i += block_dimx) {
                a[i] += b[i] + scalar * c[i];
              }
              // This or sycl::access::fence_space::local_space
              sycl::group_barrier(item.get_group());
            }

#ifdef __SYCL_DEVICE_ONLY__
            c1 = intel_get_cycle_counter();
#endif
            double clocks = (double)(c1 - c0);
            double avg_clocks = clocks / nreps;
            double data_size = (double)n * 4.0 * sizeof(double);

            if (thread_idx == 0) {
              d_bw[block_idx] = data_size / avg_clocks;
            }
          });
    }).wait();


    gpuQueue.memcpy(h_bw, d_bw, sizeof(double) * num_blocks).wait();

    // Sum the memory bw per SM to get the aggregate memory bandwidth
    for (int i = 0; i < num_blocks; ++i) {
      tot_mem_bw += h_bw[i];
    }

    sycl::free(d_a, gpuQueue);
    sycl::free(d_b, gpuQueue);
    sycl::free(d_c, gpuQueue);

    return tot_mem_bw;
  } catch (sycl::exception &e) {
    /* handle SYCL exception */
    std::cerr << e.what() << std::endl;
    return -1.0;
  }
}

#pragma once

#include <omp.h>

#ifdef __INTEL_COMPILER
#define DECLARE_ALIGNED(p, a) __assume_aligned(p, a)
#elif defined __GNUC__
#define DECLARE_ALIGNED(p, a) p = __builtin_assume_aligned(p, a)
#else
// Ignore if we're using an unsupported compiler
#define DECLARE_ALIGNED(p, a)
#endif

double cache_triad(size_t n, size_t nreps)
{
  const double scalar = 2.0f;

  double tot_mem_bw = 0.0;
#pragma omp parallel reduction(+ : tot_mem_bw)
  {
    double* restrict a = (double*)aligned_alloc(64, sizeof(double) * n);
    double* restrict b = (double*)aligned_alloc(64, sizeof(double) * n);
    double* restrict c = (double*)aligned_alloc(64, sizeof(double) * n);

    DECLARE_ALIGNED(a, 64);
    DECLARE_ALIGNED(b, 64);
    DECLARE_ALIGNED(c, 64);

    // This should place a,b,c in cache
    for (int i = 0; i < n; ++i) {
      a[i] = 0.0;
      b[i] = 3.0;
      c[i] = 2.0;
    }

    double t0 = omp_get_wtime();
    for (int t = 0; t < nreps; ++t) {
#pragma omp simd aligned(a : 64) aligned(b : 64) aligned(c : 64)
      for (int i = 0; i < n; ++i) {
        a[i] += b[i] + scalar * c[i];
      }
    }
    double t1 = omp_get_wtime();

    double mem_bw = (4.0 * sizeof(double) * n) / ((t1 - t0) / nreps) / 1e9;
    tot_mem_bw += mem_bw;

    free(a);
    free(b);
    free(c);
  }
  return tot_mem_bw;
}

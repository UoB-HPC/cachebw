#pragma once

#include "cuda_utils.h"

__global__ void triad_kernel(double* d_a, double* d_b, double* d_c, size_t n,
                             size_t nreps, double* bw, double freq)
{
  const int tid = threadIdx.x;
  const int blk_sz = blockDim.x;

  const double scalar = 2.0;

  double* __restrict__ a = d_a + blockIdx.x * n;
  double* __restrict__ b = d_b + blockIdx.x * n;
  double* __restrict__ c = d_c + blockIdx.x * n;

  // This should place a,b,c in cache 
  for (int i = tid; i < n; i += blk_sz) {
    a[i] = 0.0;
    b[i] = 3.0;
    c[i] = 2.0;
  }

  long long int c0 = clock64();
  for (int t = 0; t < nreps; ++t) {
    for (int i = tid; i < n; i += blk_sz) {
      a[i] += b[i] + scalar * c[i];
    }
    __syncthreads();
  }
  long long int c1 = clock64();

  double seconds = (((double)(c1 - c0)) / freq) / 1e9;
  double avg_seconds = seconds / nreps;
  double data_size = (double)n * 4.0 * sizeof(double) / 1e9;

  if (tid == 0) bw[blockIdx.x] = data_size / avg_seconds;
}

double cache_triad(size_t n, size_t nreps)
{
  double tot_mem_bw = 0.0;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  int num_blks = prop.multiProcessorCount * 4;
  int blk_sz = 128;

  // if we don't have enough GPU memory, don't run
  if (num_blks * n * 3 * sizeof(double) > prop.totalGlobalMem) {
    fprintf(stderr, "error: out of GPU global memory\n");
    return -1.0;
  }

  double* a;
  double* b;
  double* c;

  CUDACHK(cudaMalloc((void**)&a, sizeof(double) * n * num_blks));
  CUDACHK(cudaMalloc((void**)&b, sizeof(double) * n * num_blks));
  CUDACHK(cudaMalloc((void**)&c, sizeof(double) * n * num_blks));

  double* h_bw = (double*)malloc(sizeof(double) * num_blks);
  double* d_bw;
  CUDACHK(cudaMalloc((void**)&d_bw, sizeof(double) * num_blks));

  double freq = (double)prop.clockRate / 1e6;

  triad_kernel<<<num_blks, blk_sz>>>(a, b, c, n, nreps, d_bw, freq);
  CUDACHK(cudaGetLastError());
  CUDACHK(cudaDeviceSynchronize());

  CUDACHK(cudaMemcpy(h_bw, d_bw, sizeof(double) * num_blks,
                     cudaMemcpyDeviceToHost));

  // Sum the memory bw per SM to get the aggregate memory bandwidth
  for (int i = 0; i < num_blks; ++i) {
    tot_mem_bw += h_bw[i];
  }

  CUDACHK(cudaFree(a));
  CUDACHK(cudaFree(b));
  CUDACHK(cudaFree(c));

  return tot_mem_bw;
}

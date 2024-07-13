#pragma once

#if SHMEM==1 && (defined(SYCL_USM) || defined(SYCL_ACC))
#include "triad-sycl-shmem.hpp"
#elif SHMEM==1 && defined(CUDA)
#include "triad-cuda-shmem.h"
#elif defined(CUDA)
#include "triad-cuda.h"
#elif SHMEM==1 && defined(HIP)
#include "triad-hip-shmem.h"
#elif defined(HIP)
#include "triad-hip.hpp"
#elif defined(SYCL_USM)
#include "triad-sycl-usm.hpp"
#elif defined(SYCL_ACC)
#include "triad-sycl-acc.hpp"
#else
#include "triad-cpu.h"
#endif


#pragma once

#if SHMEM==1 && (defined(SYCL_USM) || defined(SYCL_ACC))
#include "triad-sycl-shmem.hpp"
#elif SHMEM==1
#include "triad-shmem.h"
#elif defined(GPU)
#include "triad-gpu.h"
#elif defined(SYCL_USM)
#include "triad-sycl-usm.hpp"
#elif defined(SYCL_ACC)
#include "triad-sycl-acc.hpp"
#else
#include "triad-cpu.h"
#endif


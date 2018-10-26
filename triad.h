#pragma once

#if SHMEM==1
#include "triad-shmem.h"
#elif defined(GPU)
#include "triad-gpu.h"
#else
#include "triad-cpu.h"
#endif


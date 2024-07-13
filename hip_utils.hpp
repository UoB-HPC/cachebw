#pragma once

#ifdef __HIPCC__

#define HIPCHK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void gpu_assert(hipError_t code, const char* file, int line)
{
    if (code != hipSuccess)
    {
        fprintf(stderr, "HIPCHK: %s %s %d\n", hipGetErrorString(code), file, line);
        exit(code);
    }
}

#endif


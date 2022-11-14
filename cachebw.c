#include <stdio.h>
#include <stdlib.h>
#include "triad.h"

#include "pmu.h"

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(
        stderr,
        "error: need 2 input args; (1) Size of data per thread(CPU)/SM(GPU) in KiB, (2) Number of reps\n");
    return 1;
  };
  int kbytes = atoi(argv[1]);
  int nreps = atoi(argv[2]);
  int runs = atoi(argv[3]);
  RUN_PERF(
  size_t n = ((size_t)kbytes * 1024) / (3 * sizeof(double));
  double tot_mem_bw = cache_triad(n, nreps);
  // printf("[SSTSimEng:SSTDebug] OutputLine-n %zu, reps = %d, bytes %zu, bandwidth = %f\n", n, nreps,
  //        n * 3 * sizeof(double), tot_mem_bw);
  );
  return 0;
}


    // Perf setup
  // uint64_t cycle_count = 0;
  // int grp_fd;
  // uint64_t cyc_id;

  // grp_fd = create_perf_event(0x011, -1);
  // cyc_id = get_perf_event_id(grp_fd);

  // // Perf counting
  // reset_perf_event(grp_fd, 1);
  // enable_perf_event(grp_fd, 1);

    // Perf closing
  // disable_perf_event(grp_fd, 1);
  // cycle_count = read_perf_event(grp_fd, cyc_id, 1);
  // printf("%ld-%ld\n",0x11, cycle_count);



bsub -alloc_flags gpudefault -W 02:00 -nnodes 2 -P CSC362 -env "all,LSF_CPU_ISOLATION=on" -Is /bin/bash

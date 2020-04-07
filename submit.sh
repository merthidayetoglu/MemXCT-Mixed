bsub -alloc_flags gpudefault -W 01:00 -nnodes 4 -P CSC362 -env "all,LSF_CPU_ISOLATION=on" -Is /bin/bash

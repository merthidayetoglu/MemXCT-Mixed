#!/bin/bash

#BSUB -P CSC362
#BSUB -W 00:30
#BSUB -nnodes 4
#BSUB -alloc_flags "gpudefault"
#BUSB -env "all,LSF_CPU_ISOLATION=on"
#BSUB -J shale4_comm
#BSUB -o shale4_comm.%J
#BSUB -e shale4_comm.%J

date

export NUMTHE=1501 #shale 1501 chip 1210 charcoal 4500 brain 4501
export NUMRHO=2048 #shale 2048 chip 2448 charcoal 6613 brain 11283
#DOMAIN SIZE
export STARTSLICE=0 #shale 0 (896) chip 512 (962) charcoal 0 (3815) brain 0 (5000)
export NUMSLICE=1792 #shale 1792 chip 1024 charcoal 4198 brain 9209
export BATCHPROC=2
export BATCHSIZE=256 #MUST BE MULTIPLE OF FFACTOR!!! #shale 256 chip 32
export IOBATCHSIZE=8
#DOMAIN INFORMATION
export PIXSIZE=1
export XSTART=-1024 #shale -1024 chip -1224 charcoal -3306.5 brain 5641.5
export RHOSTART=-1024
#chip -1208
#charcoal     [0,997): -3316
#          [997,1994): -3324
#         [1994,2989): -3333
#         [2989,rest): -3336
#brain    [0, 2380): 5964
#      [2380, 3195): 5961.5
#      [3195, 4010): 5964
#      [4010, 4825): 5966.5
#      [4825, 5640): 5969
#      [5640, 6455): 5961.5
#      [6455, 7270): 5976.5
#      [7270, 8085): 5961.5
#      [8085, rest): 5961.5 
#SOLVER DATA
export NUMITER=30
#TILE SIZE (MUST BE POWER OF TWO)
export SPATSIZE=128
export SPECSIZE=128
#BLOCK & BUFFER SIZE
export PROJBLOCK=1024
export BACKBLOCK=1024
export PROJBUFF=96 #KB
export BACKBUFF=96 #KB

#export SINFILE=/gpfs/alpine/scratch/merth/csc362/MemXCT_datasets/flatcorr_2x_extracted.9209s.sino.spectral.data
#export THEFILE=/gpfs/alpine/scratch/merth/csc362/MemXCT_datasets/flatcorr_2x_extracted.9209s.sino.theta.data
#export SINFILE=/gpfs/alpine/scratch/merth/csc362/MemXCT_datasets/tomo_00078_extracted.4198s.sino.spectral.data
#export THEFILE=/gpfs/alpine/scratch/merth/csc362/MemXCT_datasets/tomo_00078_extracted.4198s.sino.theta.data
export SINFILE=/gpfs/alpine/scratch/merth/csc362/MemXCT_datasets/tomo_00001_extracted.1792s.spectral.data
export THEFILE=/gpfs/alpine/scratch/merth/csc362/MemXCT_datasets/tomo_00001_extracted.1792s.theta.data
#export SINFILE=/gpfs/alpine/scratch/merth/csc362/MemXCT_datasets/tomo_chip_extracted.2048s.sino.spectral.data
#export THEFILE=/gpfs/alpine/scratch/merth/csc362/MemXCT_datasets/tomo_chip_extracted.2048s.sino.theta.data
export OUTFILE=/gpfs/alpine/scratch/merth/csc362/recon_shale.bin

export PROCPERNODE=6 #PROCS PER NODE
export PROCPERSOCKET=3 #PROCS PER SOCKET

#jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info nvprof -o /gpfs/alpine/scratch/merth/csc362/profile/timeline_%p.nvvp -f ./memxct
#mv /gpfs/alpine/scratch/merth/csc362/profile/timeline_*.nvvp .
#jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info nvprof --analysis-metrics -o /gpfs/alpine/scratch/merth/csc362/profile/analysis_%p.nvvp -f ./memxct
#mv /gpfs/alpine/scratch/merth/csc362/profile/analysis_*.nvvp .

#jsrun --smpiargs="-gpu" -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct

exit 1

module load cuda
echo DOUBLE PRECSISION PERFORMANCE
export BATCHPROC=1
cp var_double_ffactor1 vars.h
make clean
make -j
export PROCPERNODE=1 #PROCS PER NODE
export PROCPERSOCKET=1 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_double_ffactor16 vars.h
make clean
make -j
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export PROCPERNODE=6 #PROCS PER NODE
export PROCPERSOCKET=3 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_double_ffactor16_overlapped vars.h
make clean
make -j
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
echo SINGLE PRECSISION PERFORMANCE
export BATCHPROC=2
cp var_float_ffactor1 vars.h
make clean
make -j
export PROCPERNODE=1 #PROCS PER NODE
export PROCPERSOCKET=1 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_float_ffactor16 vars.h
make clean
make -j
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export PROCPERNODE=6 #PROCS PER NODE
export PROCPERSOCKET=3 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_float_ffactor16_overlapped vars.h
make clean
make -j
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
echo MIXED PRECSISION PERFORMANCE
export BATCHPROC=4
cp var_mixed_ffactor1 vars.h
make clean
make -j
export PROCPERNODE=1 #PROCS PER NODE
export PROCPERSOCKET=1 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_mixed_ffactor16 vars.h
make clean
make -j
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export PROCPERNODE=6 #PROCS PER NODE
export PROCPERSOCKET=3 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_mixed_ffactor16_overlapped vars.h
make clean
make -j
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct

exit 1

export NUMSLICE=128
export BATCHSIZE=16
module load cuda
cp var_mixed_ffactor16 vars.h
make clean
make -j
sleep 1
export SPATSIZE=256
export SPECSIZE=256
export BATCHPROC=1
jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export BATCHPROC=2
jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export BATCHPROC=4
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export BATCHPROC=8
jsrun -n8 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_mixed_ffactor8 vars.h
make clean
make -j
sleep 1
export BATCHSIZE=8
export BATCHPROC=16
jsrun -n16 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_mixed_ffactor4 vars.h
make clean
make -j
sleep 1
export BATCHSIZE=4
export BATCHPROC=32
jsrun -n32 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_mixed_ffactor2 vars.h
make clean
make -j
sleep 1
export BATCHSIZE=2
export BATCHPROC=64
jsrun -n64 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_mixed_ffactor1 vars.h
make clean
make -j
sleep 1
export BATCHSIZE=1
export BATCHPROC=128
jsrun -n128 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct

cp var_mixed_ffactor16 vars.h
make clean
make -j
sleep 1
export BATCHSIZE=16
export BATCHPROC=8
export SPATSIZE=128
export SPECSIZE=128
jsrun -n16 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
jsrun -n32 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export SPATSIZE=64
export SPECSIZE=64
jsrun -n64 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
jsrun -n128 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct

export BATCHPROC=1
export SPATSIZE=256
export SPECSIZE=256
jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export SPATSIZE=128
export SPECSIZE=128
jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export SPATSIZE=64
export SPECSIZE=64
jsrun -n8 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
jsrun -n16 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export SPATSIZE=32
export SPECSIZE=32
jsrun -n32 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
jsrun -n64 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export SPATSIZE=16
export SPECSIZE=16
jsrun -n128 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct

./run_brain.sh

exit 1

module load cuda
cp var_double_ffactor1 vars.h
make clean
make -j
export PROCPERNODE=1 #PROCS PER NODE
export PROCPERSOCKET=1 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_double_ffactor16 vars.h
make clean
make -j
export PROCPERNODE=1 #PROCS PER NODE
export PROCPERSOCKET=1 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export PROCPERNODE=3 #PROCS PER NODE
export PROCPERSOCKET=3 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export PROCPERNODE=6 #PROCS PER NODE
export PROCPERSOCKET=3 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_double_ffactor16_overlapped vars.h
make clean
make -j
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_float_ffactor1 vars.h
make clean
make -j
export PROCPERNODE=1 #PROCS PER NODE
export PROCPERSOCKET=1 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_float_ffactor16 vars.h
make clean
make -j
export PROCPERNODE=1 #PROCS PER NODE
export PROCPERSOCKET=1 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export PROCPERNODE=3 #PROCS PER NODE
export PROCPERSOCKET=3 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export PROCPERNODE=6 #PROCS PER NODE
export PROCPERSOCKET=3 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_float_ffactor16_overlapped vars.h
make clean
make -j
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_mixed_ffactor1 vars.h
make clean
make -j
export PROCPERNODE=1 #PROCS PER NODE
export PROCPERSOCKET=1 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_mixed_ffactor16 vars.h
make clean
make -j
export PROCPERNODE=1 #PROCS PER NODE
export PROCPERSOCKET=1 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export PROCPERNODE=3 #PROCS PER NODE
export PROCPERSOCKET=3 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
export PROCPERNODE=6 #PROCS PER NODE
export PROCPERSOCKET=3 #PROCS PER SOCKET
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_mixed_ffactor16_overlapped vars.h
make clean
make -j
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct


exit 1

module load cuda
for i in 1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50
do
  cp var_double vars.h
  cp factor_$i factor
  make clean
  make -j
  export NUMSLICE=$i
  export BATCHSIZE=$i
  sleep 1
  jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
done
for i in 1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50
do
  cp var_float vars.h
  cp factor_$i factor
  make clean
  make -j
  export NUMSLICE=$i
  export BATCHSIZE=$i
  sleep 1
  jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
done
for i in 1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50
do
  cp var_half vars.h
  cp factor_$i factor
  make clean
  make -j
  export NUMSLICE=$i
  export BATCHSIZE=$i
  sleep 1
  jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
done
for i in 1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50
do
  cp var_mixed vars.h
  cp factor_$i factor
  make clean
  make -j
  export NUMSLICE=$i
  export BATCHSIZE=$i
  sleep 1
  jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
done

#export PROCPERNODE=1 #PROCS PER NODE
#export PROCPERSOCKET=1 #PROCS PER SOCKET
#jsrun -n1 -a1 -g1 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n8 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n16 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n32 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n64 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#export PROCPERNODE=3 #PROCS PER NODE
#export PROCPERSOCKET=3 #PROCS PER SOCKET
#jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n8 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n16 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n32 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n64 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#export PROCPERNODE=6 #PROCS PER NODE
#export PROCPERSOCKET=3 #PROCS PER SOCKET
#jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n8 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n16 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n32 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#jsrun -n64 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct

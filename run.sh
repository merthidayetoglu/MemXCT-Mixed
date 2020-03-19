#!/bin/bash

export NUMTHE=1501
export NUMRHO=2048
#DOMAIN SIZE
export NUMX=2048
export NUMY=2048
export NUMSLICE=1
export BATCHSIZE=1
#DOMAIN INFORMATION
export XSTART=-1024
export YSTART=-1024
export RHOSTART=-1024
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

export SINFILE=/gpfs/alpine/scratch/merth/csc362/MemXCT_datasets/tomo_00001_extracted.1s.spectral.data
export THEFILE=/gpfs/alpine/scratch/merth/csc362/MemXCT_datasets/tomo_00001_extracted.1s.theta.data

export PROCPERNODE=1 #PROCS PER NODE
export PROCPERSOCKET=1 #PROCS PER SOCKET

#jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info nvprof -o /gpfs/alpine/scratch/merth/csc362/profile/timeline_%p.nvvp -f ./memxct
#jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info nvprof --profile-all-processes -o /gpfs/alpine/scratch/merth/csc362/profile/timeline_%p.nvvp -f ./memxct
#jsrun -n2 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info nvprof --analysis-metrics -o /gpfs/alpine/scratch/merth/csc362/profile/analysis.nvvp -f ./memxct
#mv /gpfs/alpine/scratch/merth/csc362/profile/timeline_*.nvvp .
#mv /gpfs/alpine/scratch/merth/csc362/profile/analysis.nvvp .

jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct

#cp var_1 vars.h
#make clean
#make -j
#jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#cp var_2 vars.h
#make clean
#make -j
#jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#cp var_4 vars.h
#make clean
#make -j
#jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#cp var_8 vars.h
#make clean
#make -j
#jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#cp var_16 vars.h
#make clean
#make -j
#jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#cp var_32 vars.h
#make clean
#make -j
#jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#cp var_64 vars.h
#make clean
#make -j
#jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
#cp var_128 vars.h
#make clean
#make -j
#jsrun -n1 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct

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

#!/bin/bash

export NUMTHE=1501
export NUMRHO=2048
#DOMAIN SIZE
export NUMX=2048
export NUMSLICE=16
export BATCHSIZE=16
#DOMAIN INFORMATION
export XSTART=-1024
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

#jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info nvprof -o /gpfs/alpine/scratch/merth/csc362/profile/timeline_%p.nvvp -f ./memxct
#jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1 -bpacked:7 js_task_info nvprof --analysis-metrics -o /gpfs/alpine/scratch/merth/csc362/profile/analysis_%p.nvvp -f ./memxct
#mv /gpfs/alpine/scratch/merth/csc362/profile/timeline_*.nvvp .
#mv /gpfs/alpine/scratch/merth/csc362/profile/analysis_*.nvvp .

jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct

exit 1

cp var_1 vars.h
make clean
make -j
export NUMSLICE=1
export BATCHSIZE=1
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_2 vars.h
make clean
make -j
export NUMSLICE=2
export BATCHSIZE=2
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_4 vars.h
make clean
make -j
export NUMSLICE=4
export BATCHSIZE=4
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_6 vars.h
make clean
make -j
export NUMSLICE=6
export BATCHSIZE=6
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_8 vars.h
make clean
make -j
export NUMSLICE=8
export BATCHSIZE=8
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_10 vars.h
make clean
make -j
export NUMSLICE=10
export BATCHSIZE=10
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_12 vars.h
make clean
make -j
export NUMSLICE=12
export BATCHSIZE=12
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_14 vars.h
make clean
make -j
export NUMSLICE=14
export BATCHSIZE=14
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_16 vars.h
make clean
make -j
export NUMSLICE=16
export BATCHSIZE=16
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_18 vars.h
make clean
make -j
export NUMSLICE=18
export BATCHSIZE=18
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_20 vars.h
make clean
make -j
export NUMSLICE=20
export BATCHSIZE=20
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_22 vars.h
make clean
make -j
export NUMSLICE=22
export BATCHSIZE=22
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_24 vars.h
make clean
make -j
export NUMSLICE=24
export BATCHSIZE=24
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_26 vars.h
make clean
make -j
export NUMSLICE=26
export BATCHSIZE=26
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_28 vars.h
make clean
make -j
export NUMSLICE=28
export BATCHSIZE=28
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_30 vars.h
make clean
make -j
export NUMSLICE=30
export BATCHSIZE=30
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_32 vars.h
make clean
make -j
export NUMSLICE=32
export BATCHSIZE=32
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_34 vars.h
make clean
make -j
export NUMSLICE=34
export BATCHSIZE=34
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_36 vars.h
make clean
make -j
export NUMSLICE=36
export BATCHSIZE=36
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_38 vars.h
make clean
make -j
export NUMSLICE=38
export BATCHSIZE=38
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_40 vars.h
make clean
make -j
export NUMSLICE=40
export BATCHSIZE=40
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_42 vars.h
make clean
make -j
export NUMSLICE=42
export BATCHSIZE=42
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_44 vars.h
make clean
make -j
export NUMSLICE=44
export BATCHSIZE=44
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_46 vars.h
make clean
make -j
export NUMSLICE=46
export BATCHSIZE=46
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_48 vars.h
make clean
make -j
export NUMSLICE=48
export BATCHSIZE=48
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct
cp var_50 vars.h
make clean
make -j
export NUMSLICE=50
export BATCHSIZE=50
sleep 1
jsrun -n4 -a6 -g6 -c42 -EOMP_NUM_THREADS=7 -r1  -bpacked:7 js_task_info ./memxct

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

#!/bin/bash
date

export NUMTHE=1801 #shale 1501 chip 1210 charcoal 4500 brain 4501
export NUMRHO=4143 #shale 2048 chip 2448 charcoal 6613 brain 11283
#DOMAIN SIZE
export STARTSLICE=0 #shale 0 (896) chip 512 (962) charcoal 0 (3815) brain 0 (5000)
export NUMSLICE=256 #shale 1792 chip 1024 charcoal 4198 brain 9209
export BATCHPROC=1
export BATCHSIZE=256 #MUST BE MULTIPLE OF FFACTOR!!! #shale 256 chip 32
export IOBATCHSIZE=256
#DOMAIN INFORMATION
export PIXSIZE=1
export XSTART=-2071.5 #shale -1024 chip -1224 charcoal -3306.5 brain 5641.5
export RHOSTART=-2083
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
export PROJBUFF=81 #KB
export BACKBUFF=81 #KB

export SINFILE=/lus/theta-fs0/projects/hp-ptycho/bicer/tomography/andrew/B1000/extracted/v1_extracted.i0.s256.spectral.data
export THEFILE=/lus/theta-fs0/projects/hp-ptycho/mert/andrew/extracted.theta.data
export OUTFILE=/lus/theta-fs0/projects/hp-ptycho/mert/v1_extracted.i0.s256.recon
#export SINFILE=/lus/theta-fs0/projects/hp-ptycho/mert/ADS3_2slice.bin
#export THEFILE=/lus/theta-fs0/projects/hp-ptycho/mert/ADS3_theta.bin
#export OUTFILE=/lus/theta-fs0/projects/hp-ptycho/mert/ADS3_recon.bin

export PROCPERNODE=1 #PROCS PER NODE
export PROCPERSOCKET=1 #PROCS PER SOCKET

#export OMP_NUM_THREADS=32
#mpirun -n 1 ncu --section MemoryWorkloadAnalysis --section SpeedOfLight memxct
#mpirun -n 4 ./memxct
#exit 1

# number of nodes
NODES=`cat $COBALT_NODEFILE | wc -l`
# processes per node
PPN=8
# total process count
PROCS=$((NODES * PPN))
# run /path/to/app.exe on $NODES nodes with $PPN processes per node
mpirun -hostfile $COBALT_NODEFILE -n $PROCS -npernode $PPN -x OMP_NUM_THREADS=32 memxct


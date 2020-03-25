#include "vars.h"

//TIMERS & COUNTERS
extern int numproj;
extern double ptime;
extern double pktime;
extern double pcstime;
extern double pcntime;
extern double pcrtime;
extern double pchtime;
extern double pmtime;
extern double prtime;
extern int numback;
extern double btime;
extern double bktime;
extern double bcstime;
extern double bcntime;
extern double bcrtime;
extern double bchtime;
extern double bmtime;
extern double brtime;

extern int raynuminc;
extern int raynumout;
extern int mynumray;
extern int mynumpix;
extern int batchsize;

extern int *raysendstart;
extern int *rayrecvstart;
extern int *raysendcount;
extern int *rayrecvcount;

extern int *rayraystart;
extern int *rayrayind;
extern int *rayrecvlist;

extern int proj_blocksize;
extern int proj_numblocks;
extern int proj_numbufftot;
extern int *proj_buffdispl;
extern int proj_buffsize;
extern int proj_mapnztot;
extern int *proj_mapdispl;
extern int *proj_mapnz;
extern int *proj_buffmap;
extern int proj_warpnztot;
extern int *proj_warpdispl;
extern unsigned short *proj_warpindex;
extern MATPREC *proj_warpvalue;

extern int back_blocksize;
extern int back_numblocks;
extern int back_numbufftot;
extern int *back_buffdispl;
extern int back_buffsize;
extern int back_mapnztot;
extern int *back_mapdispl;
extern int *back_mapnz;
extern int *back_buffmap;
extern int back_warpnztot;
extern int *back_warpdispl;
extern unsigned short *back_warpindex;
extern MATPREC *back_warpvalue;

int *proj_buffdispl_d;
int *proj_mapdispl_d;
int *proj_mapnz_d;
int *proj_buffmap_d;
int *proj_warpdispl_d;
unsigned short *proj_warpindex_d;
MATPREC *proj_warpvalue_d;
int *back_buffdispl_d;
int *back_mapdispl_d;
int *back_mapnz_d;
int *back_buffmap_d;
int *back_warpdispl_d;
unsigned short *back_warpindex_d;
MATPREC *back_warpvalue_d;

int *rayraystart_d;
int *rayrayind_d;
int *rayindray_d;


extern int socketrayout;
extern int socketrayinc;
extern int *socketreduceout;
extern int *socketreduceinc;
extern int *socketreduceoutdispl;
extern int *socketreduceincdispl;
extern int *socketsendcomm;
extern int *socketrecvcomm;
extern int *socketsendcommdispl;
extern int *socketrecvcommdispl;
extern int *socketsendmap;
extern int *socketreducedispl;
extern int *socketreduceindex;
extern int *socketraydispl;
extern int *socketrayindex;
extern int *socketpackmap;
extern int *socketunpackmap;

extern int noderayout;
extern int noderayinc;
extern int *nodereduceout;
extern int *nodereduceinc;
extern int *nodereduceoutdispl;
extern int *nodereduceincdispl;
extern int *nodesendcomm;
extern int *noderecvcomm;
extern int *nodesendcommdispl;
extern int *noderecvcommdispl;
extern int *nodesendmap;
extern int *nodereducedispl;
extern int *nodereduceindex;
extern int *noderaydispl;
extern int *noderayindex;
extern int *nodepackmap;
extern int *nodeunpackmap;

extern int *raypackmap;
extern int *rayunpackmap;

extern int numthreads;
extern int numproc;
extern int myid;
extern MPI_Comm MPI_COMM_SOCKET;
extern int numproc_socket;
extern int myid_socket;
extern int numsocket;
extern MPI_Comm MPI_COMM_NODE;
extern int numproc_node;
extern int myid_node;
extern int numnode;

int *socketpackmap_d;
int *socketunpackmap_d;
int *socketreducedispl_d;
int *socketreduceindex_d;
int *nodepackmap_d;
int *nodeunpackmap_d;
int *nodereducedispl_d;
int *nodereduceindex_d;
int *raypackmap_d;
int *rayunpackmap_d;
int *noderaydispl_d;
int *noderayindex_d;

VECPREC *socketreducesendbuff_d;
VECPREC *socketreducerecvbuff_d;
VECPREC *nodereducesendbuff_d;
VECPREC *nodereducerecvbuff_d;
VECPREC *nodesendbuff_d;
VECPREC *noderecvbuff_d;
VECPREC *nodesendbuff_h;
VECPREC *noderecvbuff_h;

extern int *socketrecvbuffdispl_p;
extern VECPREC **socketrecvbuff_p;
extern int *socketrecvdevice_p;
extern int *noderecvbuffdispl_p;
extern VECPREC **noderecvbuff_p;
extern int *noderecvdevice_p;

__global__ void kernel_project __launch_bounds__(1024,1) (VECPREC*,double*,unsigned short*,MATPREC*,int,int,int*,int*,int*,int*,int*,int,int*);
__global__ void kernel_backproject __launch_bounds__(1024,1) (double*,VECPREC*,unsigned short*,MATPREC*,int,int,int*,int*,int*,int*,int*,int,int*);
__global__ void kernel_reduce(VECPREC*,VECPREC*,int*,int*,int,int,int*,int*);
__global__ void kernel_reducenopack(double*,VECPREC*,int*,int*,int,int,int*);
__global__ void kernel_scatternopack(double*,VECPREC*,int*,int*,int,int,int*);
__global__ void kernel_scatter(VECPREC*,VECPREC*,int*,int*,int,int,int*,int*);

int numdevice;
int mydevice;
cudaEvent_t start,stop;
float milliseconds;
MPI_Request *sendrequest;
MPI_Request *recvrequest;
cudaStream_t *socketstream;
cudaStream_t *nodestream;

void setup_gpu(double **obj, double **gra, double **dir, double **mes, double **res, double **ray){

  cudaGetDeviceCount(&numdevice);
  mydevice = myid%numdevice;
  cudaSetDevice(mydevice);
  if(myid==0){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("\n");
    printf("Device Count: %d\n",deviceCount);
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d name: %s\n",dev,deviceProp.name);
    printf("Computational Capabilities: %d, %d\n",deviceProp.major,deviceProp.minor);
    printf("Maximum global memory size: %lu\n",deviceProp.totalGlobalMem);
    printf("Maximum constant memory size: %lu\n",deviceProp.totalConstMem);
    printf("Maximum shared memory size per block: %lu\n",deviceProp.sharedMemPerBlock);
    printf("Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf("Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    printf("Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
    printf("Warp size: %d\n",deviceProp.warpSize);
    printf("\n");
  }

  //CONJUGATE-GRADIENT BUFFERS
  double batchmem = 0.0;
  batchmem += sizeof(double)*mynumpix*batchsize/1.0e9;
  batchmem += sizeof(double)*mynumpix*batchsize/1.0e9;
  batchmem += sizeof(double)*mynumpix*batchsize/1.0e9;
  batchmem += sizeof(double)*mynumray*batchsize/1.0e9;
  batchmem += sizeof(double)*mynumray*batchsize/1.0e9;
  batchmem += sizeof(double)*mynumray*batchsize/1.0e9;
  cudaMalloc((void**)obj,sizeof(double)*mynumpix*batchsize);
  cudaMalloc((void**)gra,sizeof(double)*mynumpix*batchsize);
  cudaMalloc((void**)dir,sizeof(double)*mynumpix*batchsize);
  cudaMalloc((void**)mes,sizeof(double)*mynumray*batchsize);
  cudaMalloc((void**)res,sizeof(double)*mynumray*batchsize);
  cudaMalloc((void**)ray,sizeof(double)*mynumray*batchsize);
  //COMMUNICATION BUFFERS
  double commem = 0.0;
  commem += sizeof(VECPREC)*socketsendcommdispl[numproc_socket]*FFACTOR/1.0e9;
  commem += sizeof(VECPREC)*socketrecvcommdispl[numproc_socket]*FFACTOR/1.0e9;
  commem += sizeof(VECPREC)*nodesendcommdispl[numproc_node]*FFACTOR/1.0e9;
  commem += sizeof(VECPREC)*noderecvcommdispl[numproc_node]*FFACTOR/1.0e9;
  commem += sizeof(VECPREC)*nodereduceoutdispl[numproc]*FFACTOR/1.0e9;
  commem += sizeof(VECPREC)*nodereduceincdispl[numproc]*FFACTOR/1.0e9;
  cudaMalloc((void**)&socketreducesendbuff_d,sizeof(VECPREC)*socketsendcommdispl[numproc_socket]*FFACTOR);
  cudaMalloc((void**)&socketreducerecvbuff_d,sizeof(VECPREC)*socketrecvcommdispl[numproc_socket]*FFACTOR);
  cudaMalloc((void**)&nodereducesendbuff_d,sizeof(VECPREC)*nodesendcommdispl[numproc_node]*FFACTOR);
  cudaMalloc((void**)&nodereducerecvbuff_d,sizeof(VECPREC)*noderecvcommdispl[numproc_node]*FFACTOR);
  cudaMalloc((void**)&nodesendbuff_d,sizeof(VECPREC)*nodereduceoutdispl[numproc]*FFACTOR);
  cudaMalloc((void**)&noderecvbuff_d,sizeof(VECPREC)*nodereduceincdispl[numproc]*FFACTOR);
  //PACK AND UNPACK MAPS
  commem += sizeof(int)*socketsendcommdispl[numproc_socket]*FFACTOR/1.0e9;
  commem += sizeof(int)*socketrecvcommdispl[numproc_socket]*FFACTOR/1.0e9;
  commem += sizeof(int)*(socketreduceoutdispl[numproc]+1)/1.0e9;
  commem += sizeof(int)*socketreducedispl[socketreduceoutdispl[numproc]]/1.0e9;
  commem += sizeof(int)*socketreduceoutdispl[numproc]*FFACTOR/1.0e9;
  commem += sizeof(int)*noderecvcommdispl[numproc_node]*FFACTOR/1.0e9;
  commem += sizeof(int)*(nodereduceoutdispl[numproc]+1)/1.0e9;
  commem += sizeof(int)*nodereducedispl[nodereduceoutdispl[numproc]]/1.0e9;
  commem += sizeof(int)*nodereduceoutdispl[numproc]*FFACTOR/1.0e9;
  commem += sizeof(int)*nodereduceincdispl[numproc]*FFACTOR/1.0e9;
  commem += sizeof(int)*(mynumray+1)/1.0e9;
  commem += sizeof(int)*noderaydispl[mynumray]/1.0e9;
  cudaMalloc((void**)&socketpackmap_d,sizeof(int)*socketsendcommdispl[numproc_socket]*FFACTOR);
  cudaMalloc((void**)&socketunpackmap_d,sizeof(int)*socketrecvcommdispl[numproc_socket]*FFACTOR);
  cudaMalloc((void**)&socketreducedispl_d,sizeof(int)*(socketreduceoutdispl[numproc]+1));
  cudaMalloc((void**)&socketreduceindex_d,sizeof(int)*socketreducedispl[socketreduceoutdispl[numproc]]);
  cudaMalloc((void**)&nodepackmap_d,sizeof(int)*socketreduceoutdispl[numproc]*FFACTOR);
  cudaMalloc((void**)&nodeunpackmap_d,sizeof(int)*noderecvcommdispl[numproc_node]*FFACTOR);
  cudaMalloc((void**)&nodereducedispl_d,sizeof(int)*(nodereduceoutdispl[numproc]+1));
  cudaMalloc((void**)&nodereduceindex_d,sizeof(int)*nodereducedispl[nodereduceoutdispl[numproc]]);
  cudaMalloc((void**)&raypackmap_d,sizeof(int)*nodereduceoutdispl[numproc]*FFACTOR);
  cudaMalloc((void**)&rayunpackmap_d,sizeof(int)*nodereduceincdispl[numproc]*FFACTOR);
  cudaMalloc((void**)&noderaydispl_d,sizeof(int)*(mynumray+1));
  cudaMalloc((void**)&noderayindex_d,sizeof(int)*noderaydispl[mynumray]);
  cudaMemcpy(socketpackmap_d,socketpackmap,sizeof(int)*socketsendcommdispl[numproc_socket]*FFACTOR,cudaMemcpyHostToDevice);
  cudaMemcpy(socketunpackmap_d,socketunpackmap,sizeof(int)*socketrecvcommdispl[numproc_socket]*FFACTOR,cudaMemcpyHostToDevice);
  cudaMemcpy(socketreducedispl_d,socketreducedispl,sizeof(int)*(socketreduceoutdispl[numproc]+1),cudaMemcpyHostToDevice);
  cudaMemcpy(socketreduceindex_d,socketreduceindex,sizeof(int)*socketreducedispl[socketreduceoutdispl[numproc]],cudaMemcpyHostToDevice);
  cudaMemcpy(nodepackmap_d,nodepackmap,sizeof(int)*socketreduceoutdispl[numproc]*FFACTOR,cudaMemcpyHostToDevice);
  cudaMemcpy(nodeunpackmap_d,nodeunpackmap,sizeof(int)*noderecvcommdispl[numproc_node]*FFACTOR,cudaMemcpyHostToDevice);
  cudaMemcpy(nodereducedispl_d,nodereducedispl,sizeof(int)*(nodereduceoutdispl[numproc]+1),cudaMemcpyHostToDevice);
  cudaMemcpy(nodereduceindex_d,nodereduceindex,sizeof(int)*nodereducedispl[nodereduceoutdispl[numproc]],cudaMemcpyHostToDevice);
  cudaMemcpy(raypackmap_d,raypackmap,sizeof(int)*nodereduceoutdispl[numproc]*FFACTOR,cudaMemcpyHostToDevice);
  cudaMemcpy(rayunpackmap_d,rayunpackmap,sizeof(int)*nodereduceincdispl[numproc]*FFACTOR,cudaMemcpyHostToDevice);
  cudaMemcpy(noderaydispl_d,noderaydispl,sizeof(int)*(mynumray+1),cudaMemcpyHostToDevice);
  cudaMemcpy(noderayindex_d,noderayindex,sizeof(int)*noderaydispl[mynumray],cudaMemcpyHostToDevice);
 
  cudaMallocHost((void**)&nodesendbuff_h,sizeof(VECPREC)*nodereduceoutdispl[numproc]*FFACTOR);
  cudaMallocHost((void**)&noderecvbuff_h,sizeof(VECPREC)*nodereduceincdispl[numproc]*FFACTOR);

  double projmem = 0.0;
  projmem = projmem + sizeof(int)/1.0e9*(proj_numblocks+1);
  projmem = projmem + sizeof(int)/1.0e9*(proj_numbufftot+1);
  projmem = projmem + sizeof(int)/1.0e9*proj_numbufftot;
  projmem = projmem + sizeof(int)/1.0e9*proj_mapnztot;
  projmem = projmem + sizeof(int)/1.0e9*(proj_numbufftot*(proj_blocksize/WARPSIZE)+1);
  projmem = projmem + sizeof(unsigned short)/1.0e9*(proj_warpnztot*WARPSIZE);
  projmem = projmem + sizeof(MATPREC)/1.0e9*(proj_warpnztot*WARPSIZE);
  projmem = projmem + sizeof(int)/1.0e9*proj_mapnztot;
  //printf("PROC %d FORWARD PROJECTION MEMORY: %f GB\n",myid,projmem);

  cudaMalloc((void**)&proj_buffdispl_d,sizeof(int)*(proj_numblocks+1));
  cudaMalloc((void**)&proj_mapdispl_d,sizeof(int)*(proj_numbufftot+1));
  cudaMalloc((void**)&proj_mapnz_d,sizeof(int)*proj_numbufftot);
  cudaMalloc((void**)&proj_buffmap_d,sizeof(int)*proj_mapnztot);
  cudaMalloc((void**)&proj_warpdispl_d,sizeof(int)*(proj_numbufftot*(proj_blocksize/WARPSIZE)+1));
  cudaMalloc((void**)&proj_warpindex_d,sizeof(unsigned short)*proj_warpnztot*WARPSIZE);
  cudaMalloc((void**)&proj_warpvalue_d,sizeof(MATPREC)*proj_warpnztot*WARPSIZE);
  cudaMemcpy(proj_buffdispl_d,proj_buffdispl,sizeof(int)*(proj_numblocks+1),cudaMemcpyHostToDevice);
  cudaMemcpy(proj_mapdispl_d,proj_mapdispl,sizeof(int)*(proj_numbufftot+1),cudaMemcpyHostToDevice);
  cudaMemcpy(proj_mapnz_d,proj_mapnz,sizeof(int)*proj_numbufftot,cudaMemcpyHostToDevice);
  cudaMemcpy(proj_buffmap_d,proj_buffmap,sizeof(int)*proj_mapnztot,cudaMemcpyHostToDevice);
  cudaMemcpy(proj_warpdispl_d,proj_warpdispl,sizeof(int)*(proj_numbufftot*(proj_blocksize/WARPSIZE)+1),cudaMemcpyHostToDevice);
  cudaMemcpy(proj_warpindex_d,proj_warpindex,sizeof(unsigned short)*proj_warpnztot*WARPSIZE,cudaMemcpyHostToDevice);
  cudaMemcpy(proj_warpvalue_d,proj_warpvalue,sizeof(MATPREC)*proj_warpnztot*WARPSIZE,cudaMemcpyHostToDevice);

  cudaMalloc((void**)&back_buffdispl_d,sizeof(int)*(back_numblocks+1));
  cudaMalloc((void**)&back_mapdispl_d,sizeof(int)*(back_numbufftot+1));
  cudaMalloc((void**)&back_mapnz_d,sizeof(int)*back_numbufftot);
  cudaMalloc((void**)&back_buffmap_d,sizeof(int)*back_mapnztot);
  cudaMalloc((void**)&back_warpdispl_d,sizeof(int)*(back_numbufftot*(back_blocksize/WARPSIZE)+1));
  cudaMalloc((void**)&back_warpindex_d,sizeof(unsigned short)*back_warpnztot*WARPSIZE);
  cudaMalloc((void**)&back_warpvalue_d,sizeof(MATPREC)*back_warpnztot*WARPSIZE);
  cudaMemcpy(back_buffdispl_d,back_buffdispl,sizeof(int)*(back_numblocks+1),cudaMemcpyHostToDevice);
  cudaMemcpy(back_mapdispl_d,back_mapdispl,sizeof(int)*(back_numbufftot+1),cudaMemcpyHostToDevice);
  cudaMemcpy(back_mapnz_d,back_mapnz,sizeof(int)*back_numbufftot,cudaMemcpyHostToDevice);
  cudaMemcpy(back_buffmap_d,back_buffmap,sizeof(int)*back_mapnztot,cudaMemcpyHostToDevice);
  cudaMemcpy(back_warpdispl_d,back_warpdispl,sizeof(int)*(back_numbufftot*(back_blocksize/WARPSIZE)+1),cudaMemcpyHostToDevice);
  cudaMemcpy(back_warpindex_d,back_warpindex,sizeof(unsigned short)*back_warpnztot*WARPSIZE,cudaMemcpyHostToDevice);
  cudaMemcpy(back_warpvalue_d,back_warpvalue,sizeof(MATPREC)*back_warpnztot*WARPSIZE,cudaMemcpyHostToDevice);

  double backmem = 0.0;
  backmem = backmem + sizeof(int)/1.0e9*(back_numblocks+1);
  backmem = backmem + sizeof(int)/1.0e9*(back_numbufftot+1);
  backmem = backmem + sizeof(int)/1.0e9*back_numbufftot;
  backmem = backmem + sizeof(int)/1.0e9*back_mapnztot;
  backmem = backmem + sizeof(int)/1.0e9*(back_numbufftot*(back_blocksize/WARPSIZE)+1);
  backmem = backmem + sizeof(unsigned short)/1.0e9*(back_warpnztot*WARPSIZE);
  backmem = backmem + sizeof(MATPREC)/1.0e9*(back_warpnztot*WARPSIZE);
  backmem = backmem + sizeof(int)/1.0e9*back_mapnztot;
  //printf("PROC %d BACKPROJECTION MEMORY: %f GB\n",myid,backmem);

  double gpumem = projmem+backmem;
  double gpumems[numproc];
  double batchmems[numproc];
  double commems[numproc];
  MPI_Allgather(&gpumem,1,MPI_DOUBLE,gpumems,1,MPI_DOUBLE,MPI_COMM_WORLD);
  MPI_Allgather(&batchmem,1,MPI_DOUBLE,batchmems,1,MPI_DOUBLE,MPI_COMM_WORLD);
  MPI_Allgather(&commem,1,MPI_DOUBLE,commems,1,MPI_DOUBLE,MPI_COMM_WORLD);
  if(myid==0){
    double gputotmem = 0.0;
    double batchtotmem = 0.0;
    double commtotmem = 0.0;
    for(int p = 0; p < numproc; p++){
      printf("PROC %d GPU MEMORY: %f GB + %f GB + %f GB = %f GB\n",p,gpumems[p],batchmems[p],commems[p],gpumems[p]+batchmems[p]+commems[p]);
      gputotmem += gpumems[p];
      batchtotmem += batchmems[p];
      commtotmem += commems[p];
    }
    printf("TOTAL GPU MEMORY %f GB + %f GB + %f GB = %f GB\n",gputotmem,batchtotmem,commtotmem,gputotmem+batchtotmem+commtotmem);
  }

  cudaFuncSetAttribute(kernel_project,cudaFuncAttributeMaxDynamicSharedMemorySize,98304);
  cudaFuncSetAttribute(kernel_backproject,cudaFuncAttributeMaxDynamicSharedMemorySize,98304);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  sendrequest = new MPI_Request[numproc];
  recvrequest = new MPI_Request[numproc];
  socketstream = new cudaStream_t[numproc_socket];
  nodestream = new cudaStream_t[numproc_node];
  for(int p = 0; p < numproc_socket; p++)
    cudaStreamCreate(&socketstream[p]);
  for(int p = 0; p < numproc_node; p++)
    cudaStreamCreate(&nodestream[p]);

  communications();

}

void projection(double *sino_d, double *tomo_d){
  MPI_Barrier(MPI_COMM_WORLD);
  double projectiontime = MPI_Wtime();
  for(int slice = 0; slice < batchsize; slice += FFACTOR){
    //PARTIAL PROJECTION
    cudaEventRecord(start);
    kernel_project<<<proj_numblocks,proj_blocksize,sizeof(VECPREC)*proj_buffsize*FFACTOR>>>(socketreducesendbuff_d,tomo_d+slice*mynumpix,proj_warpindex_d,proj_warpvalue_d,raynumout,mynumpix,proj_buffdispl_d,proj_warpdispl_d,proj_mapdispl_d,proj_mapnz_d,proj_buffmap_d,proj_buffsize,socketpackmap_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    //if(myid==0)printf("project %e milliseconds\n",milliseconds);
    pktime += milliseconds/1e3;
    //SOCKET COMMUNICATION
    MPI_Barrier(MPI_COMM_SOCKET);
    double cstime = MPI_Wtime();
    for(int psend = 0; psend < numproc_socket; psend++)
      if(socketsendcomm[psend])
        cudaMemcpyPeerAsync(socketrecvbuff_p[psend]+socketrecvbuffdispl_p[psend]*FFACTOR,socketrecvdevice_p[psend],socketreducesendbuff_d+socketsendcommdispl[psend]*FFACTOR,mydevice,sizeof(VECPREC)*socketsendcomm[psend]*FFACTOR,socketstream[psend]);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_SOCKET);
    //if(myid==0)printf("socket time %e\n",MPI_Wtime()-cstime);
    pcstime += MPI_Wtime()-cstime;
    //SOCKET REDUCTION
    cudaEventRecord(start);
    kernel_reduce<<<(socketreduceoutdispl[numproc]+255)/256,256>>>(nodereducesendbuff_d,socketreducerecvbuff_d,socketreducedispl_d,socketreduceindex_d,socketreduceoutdispl[numproc],socketrecvcommdispl[numproc_socket],nodepackmap_d,socketunpackmap_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    prtime += milliseconds/1e3;
    //NODE COMMUNICATION
    MPI_Barrier(MPI_COMM_NODE);
    double cntime = MPI_Wtime();
    for(int psend = 0; psend < numproc_node; psend++)
      if(nodesendcomm[psend])
        cudaMemcpyPeerAsync(noderecvbuff_p[psend]+noderecvbuffdispl_p[psend]*FFACTOR,noderecvdevice_p[psend],nodereducesendbuff_d+nodesendcommdispl[psend]*FFACTOR,mydevice,sizeof(VECPREC)*nodesendcomm[psend]*FFACTOR,nodestream[psend]);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_NODE);
    //if(myid==0)printf("node time %e\n",MPI_Wtime()-cntime);
    pcntime += MPI_Wtime()-cntime;
    //NODE REDUCTION
    cudaEventRecord(start);
    kernel_reduce<<<(nodereduceoutdispl[numproc]+255)/256,256>>>(nodesendbuff_d,nodereducerecvbuff_d,nodereducedispl_d,nodereduceindex_d,nodereduceoutdispl[numproc],noderecvcommdispl[numproc_node],raypackmap_d,nodeunpackmap_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    prtime += milliseconds/1e3;
    //MEMCPY DEVICE TO HOST
    cudaEventRecord(start);
    cudaMemcpy(nodesendbuff_h,nodesendbuff_d,sizeof(VECPREC)*nodereduceoutdispl[numproc]*FFACTOR,cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    pmtime += milliseconds/1e3;
    //HOST COMMUNICATION
    MPI_Barrier(MPI_COMM_WORLD);
    double chtime = MPI_Wtime();
    {
      int sendcount = 0;
      int recvcount = 0;
      for(int p = 0; p < numproc; p++){
        if(nodereduceout[p]){
          MPI_Issend(nodesendbuff_h+nodereduceoutdispl[p]*FFACTOR,nodereduceout[p]*FFACTOR*sizeof(VECPREC),MPI_BYTE,p,0,MPI_COMM_WORLD,sendrequest+sendcount);
          sendcount++;
        }
        if(nodereduceinc[p]){
          MPI_Irecv(noderecvbuff_h+nodereduceincdispl[p]*FFACTOR,nodereduceinc[p]*FFACTOR*sizeof(VECPREC),MPI_BYTE,p,0,MPI_COMM_WORLD,recvrequest+recvcount);
          recvcount++;
        }
      }
      MPI_Waitall(recvcount,recvrequest,MPI_STATUSES_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //if(myid==0)printf("rack time %e\n",MPI_Wtime()-chtime); 
    pchtime += MPI_Wtime()-chtime;
    //MEMCPY HOST TO DEVICE
    cudaEventRecord(start);
    cudaMemcpy(noderecvbuff_d,noderecvbuff_h,sizeof(VECPREC)*nodereduceincdispl[numproc]*FFACTOR,cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    pmtime += milliseconds/1e3;
    //HOST REDUCTION
    cudaEventRecord(start);
    kernel_reducenopack<<<(mynumray+255)/256,256>>>(sino_d+slice*mynumray,noderecvbuff_d,noderaydispl_d,noderayindex_d,mynumray,nodereduceincdispl[numproc],rayunpackmap_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    prtime += milliseconds/1e3;

    MPI_Barrier(MPI_COMM_WORLD);
    numproj++;
  }
  ptime += MPI_Wtime()-projectiontime;
}

void backproject(double *tomo_d, double *sino_d){
  MPI_Barrier(MPI_COMM_WORLD);
  double backprojtime = MPI_Wtime();
  for(int slice = 0; slice < batchsize; slice += FFACTOR){
    //HOST SCATTER
    cudaEventRecord(start);
    kernel_scatternopack<<<(mynumray+255)/256,256>>>(sino_d+slice*mynumray,noderecvbuff_d,noderaydispl_d,noderayindex_d,mynumray,nodereduceincdispl[numproc],rayunpackmap_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    brtime += milliseconds/1e3;
    //MEMCPY DEVICE TO HOST
    cudaEventRecord(start);
    cudaMemcpy(noderecvbuff_h,noderecvbuff_d,sizeof(VECPREC)*nodereduceincdispl[numproc]*FFACTOR,cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    bmtime += milliseconds/1e3;
    //HOST COMMUNICATION
    MPI_Barrier(MPI_COMM_WORLD);
    double chtime = MPI_Wtime();
    {
      int sendcount = 0;
      int recvcount = 0;
      for(int p = 0; p < numproc; p++){
        if(nodereduceout[p]){
          MPI_Irecv(nodesendbuff_h+nodereduceoutdispl[p]*FFACTOR,nodereduceout[p]*FFACTOR*sizeof(VECPREC),MPI_BYTE,p,0,MPI_COMM_WORLD,recvrequest+recvcount);
          recvcount++;
        }
        if(nodereduceinc[p]){
          MPI_Issend(noderecvbuff_h+nodereduceincdispl[p]*FFACTOR,nodereduceinc[p]*FFACTOR*sizeof(VECPREC),MPI_BYTE,p,0,MPI_COMM_WORLD,sendrequest+sendcount);
          sendcount++;
        }
      }
      MPI_Waitall(recvcount,recvrequest,MPI_STATUSES_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //if(myid==0)printf("rack time %e\n",MPI_Wtime()-chtime);
    bchtime += MPI_Wtime()-chtime;
    //MEMCPY HOST TO DEVICE
    cudaEventRecord(start);
    cudaMemcpy(nodesendbuff_d,nodesendbuff_h,sizeof(VECPREC)*nodereduceoutdispl[numproc]*FFACTOR,cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    bmtime += milliseconds/1e3;
    //NODE SCATTER
    cudaEventRecord(start);
    kernel_scatter<<<(nodereduceoutdispl[numproc]+255)/256,256>>>(nodesendbuff_d,nodereducerecvbuff_d,nodereducedispl_d,nodereduceindex_d,nodereduceoutdispl[numproc],noderecvcommdispl[numproc_node],raypackmap_d,nodeunpackmap_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    brtime += milliseconds/1e3;
    //NODE COMMUNICATION
    MPI_Barrier(MPI_COMM_NODE);
    double cntime = MPI_Wtime();
    for(int psend = 0; psend < numproc_node; psend++)
      if(nodesendcomm[psend])
        cudaMemcpyPeerAsync(nodereducesendbuff_d+nodesendcommdispl[psend]*FFACTOR,mydevice,noderecvbuff_p[psend]+noderecvbuffdispl_p[psend]*FFACTOR,noderecvdevice_p[psend],sizeof(VECPREC)*nodesendcomm[psend]*FFACTOR,nodestream[psend]);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_NODE);
    //if(myid==0)printf("node time %e\n",MPI_Wtime()-cntime);
    bcntime += MPI_Wtime()-cntime;
    //SOCKET SCATTER
    cudaEventRecord(start);
    kernel_scatter<<<(socketreduceoutdispl[numproc]+255)/256,256>>>(nodereducesendbuff_d,socketreducerecvbuff_d,socketreducedispl_d,socketreduceindex_d,socketreduceoutdispl[numproc],socketrecvcommdispl[numproc_socket],nodepackmap_d,socketunpackmap_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    brtime += milliseconds/1e3;
    //SOCKET COMMUNICATION
    MPI_Barrier(MPI_COMM_SOCKET);
    double cstime = MPI_Wtime();
    for(int psend = 0; psend < numproc_socket; psend++)
      if(socketsendcomm[psend])
        cudaMemcpyPeerAsync(socketreducesendbuff_d+socketsendcommdispl[psend]*FFACTOR,mydevice,socketrecvbuff_p[psend]+socketrecvbuffdispl_p[psend]*FFACTOR,socketrecvdevice_p[psend],sizeof(VECPREC)*socketsendcomm[psend]*FFACTOR,socketstream[psend]);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_SOCKET);
    //if(myid==0)printf("socket time %e\n",MPI_Wtime()-cstime);
    bcstime += MPI_Wtime()-cstime;
    //BACKPROJECTION
    cudaEventRecord(start);
    kernel_backproject<<<back_numblocks,back_blocksize,sizeof(VECPREC)*back_buffsize*FFACTOR>>>(tomo_d+slice*mynumpix,socketreducesendbuff_d,back_warpindex_d,back_warpvalue_d,mynumpix,raynumout,back_buffdispl_d,back_warpdispl_d,back_mapdispl_d,back_mapnz_d,back_buffmap_d,back_buffsize,socketpackmap_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    //if(myid==0)printf("backproject %e milliseconds\n",milliseconds);
    bktime += milliseconds/1e3;

    MPI_Barrier(MPI_COMM_WORLD);
    numback++;
  }
  btime += MPI_Wtime()-backprojtime;
}
__global__ void kernel_project(VECPREC *y, double *x, unsigned short *index, MATPREC *value, int numrow, int numcol, int *buffdispl, int *displ, int *mapdispl, int *mapnz, int *buffmap, int buffsize, int *packmap){
  extern __shared__ VECPREC shared[];
  VECPREC acc[FFACTOR] = {0.0};
  int wind = threadIdx.x%WARPSIZE;
  for(int buff = buffdispl[blockIdx.x]; buff < buffdispl[blockIdx.x+1]; buff++){
    int mapoffset = mapdispl[buff];
    for(int i = threadIdx.x; i < mapnz[buff]; i += blockDim.x){
      int ind = buffmap[mapoffset+i];
      for(int f = 0; f < FFACTOR; f++)
        shared[f*buffsize+i] = x[f*numcol+ind];
    }
    __syncthreads();
    int warp = (buff*blockDim.x+threadIdx.x)/WARPSIZE;
    for(int n = displ[warp]; n < displ[warp+1]; n++){
      unsigned short ind = index[n*WARPSIZE+wind];
      MATPREC val = value[n*WARPSIZE+wind];
      for(int f = 0; f < FFACTOR; f++)
        acc[f] += shared[f*buffsize+ind]*val;
    }
    __syncthreads();
  }
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  if(row  < numrow)
    for(int f = 0; f < FFACTOR; f++)
      y[packmap[f*numrow+row]] = acc[f];
}
__global__ void kernel_backproject(double *y, VECPREC *x, unsigned short *index, MATPREC *value, int numrow, int numcol, int *buffdispl, int *displ, int *mapdispl, int *mapnz, int *buffmap, int buffsize, int *packmap){
  extern __shared__ VECPREC shared[];
  VECPREC acc[FFACTOR] = {0.0};
  int wind = threadIdx.x%WARPSIZE;
  for(int buff = buffdispl[blockIdx.x]; buff < buffdispl[blockIdx.x+1]; buff++){
    int mapoffset = mapdispl[buff];
    for(int i = threadIdx.x; i < mapnz[buff]; i += blockDim.x){
      int ind = buffmap[mapoffset+i];
      for(int f = 0; f < FFACTOR; f++)
        shared[f*buffsize+i] = x[packmap[f*numcol+ind]];
    }
    __syncthreads();
    int warp = (buff*blockDim.x+threadIdx.x)/WARPSIZE;
    for(int n = displ[warp]; n < displ[warp+1]; n++){
      unsigned short ind = index[n*WARPSIZE+wind];
      MATPREC val = value[n*WARPSIZE+wind];
      for(int f = 0; f < FFACTOR; f++)
        acc[f] += shared[f*buffsize+ind]*val;
    }
    __syncthreads();
  }
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  if(row  < numrow)
    for(int f = 0; f < FFACTOR; f++)
      y[f*numrow+row] = acc[f];
}
__global__ void kernel_reduce(VECPREC *y, VECPREC *x, int *displ, int *index, int numrow, int numcol, int *packmap, int *unpackmap){
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  VECPREC reduce[FFACTOR] = {0.0};
  if(row < numrow){
    for(int n = displ[row]; n < displ[row+1]; n++){
      int ind = index[n];
      for(int f = 0; f < FFACTOR; f++)
        reduce[f] += x[unpackmap[f*numcol+ind]];
    }
    for(int f = 0; f < FFACTOR; f++)
      y[packmap[f*numrow+row]] = reduce[f];
  }
}
__global__ void kernel_reducenopack(double *y, VECPREC *x, int *displ, int *index, int numrow, int numcol, int *unpackmap){
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  VECPREC reduce[FFACTOR] = {0.0};
  if(row < numrow){
    for(int n = displ[row]; n < displ[row+1]; n++){
      int ind = index[n];
      for(int f = 0; f < FFACTOR; f++)
        reduce[f] += x[unpackmap[f*numcol+ind]];
    }
    for(int f = 0; f < FFACTOR; f++)
      y[f*numrow+row] = reduce[f];
  }
}
__global__ void kernel_scatternopack(double *y, VECPREC *x, int *displ, int *index, int numrow, int numcol, int *unpackmap){
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  VECPREC scatter[FFACTOR] = {0.0};
  if(row < numrow){
    for(int f = 0; f < FFACTOR; f++)
      scatter[f] = y[f*numrow+row];
    for(int n = displ[row]; n < displ[row+1]; n++){
      int ind = index[n];
      for(int f = 0; f < FFACTOR; f++)
        x[unpackmap[f*numcol+ind]] = scatter[f];
    }
  }
}
__global__ void kernel_scatter(VECPREC *y, VECPREC *x, int *displ, int *index, int numrow, int numcol, int *packmap, int *unpackmap){
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  VECPREC scatter[FFACTOR] = {0.0};
  if(row < numrow){
    for(int f = 0; f < FFACTOR; f++)
      scatter[f] = y[packmap[f*numrow+row]];
    for(int n = displ[row]; n < displ[row+1]; n++){
      int ind = index[n];
      for(int f = 0; f < FFACTOR; f++)
        x[unpackmap[f*numcol+ind]] = scatter[f];
    }
  }
}
void copy_kernel(double *a, double *b, int dim){
  cudaMemcpy(a,b,sizeof(double)*dim,cudaMemcpyDeviceToDevice);
};
__global__ void kernel_saxpy(double *a, double *b, double coef, double *c, int dim){
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid < dim)
    a[tid] = b[tid] + coef*c[tid];
};
void saxpy_kernel(double *a, double *b, double coef, double *c, int dim){
  kernel_saxpy<<<(dim+255)/256,256>>>(a,b,coef,c,dim);
};
__global__ void kernel_scale(double *a,  double coef, int dim){
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid < dim)
    a[tid] = coef*a[tid];
};
void scale_kernel(double *a, double coef, int dim){
  kernel_scale<<<(dim+255)/256,256>>>(a,coef,dim);
};

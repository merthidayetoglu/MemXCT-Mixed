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
#ifdef MATRIX
extern matrix *proj_warpindval;
extern matrix *back_warpindval;
#else
extern unsigned short *proj_warpindex;
extern MATPREC *proj_warpvalue;
extern unsigned short *back_warpindex;
extern MATPREC *back_warpvalue;
#endif

int *proj_buffdispl_d;
int *proj_mapdispl_d;
int *proj_mapnz_d;
int *proj_buffmap_d;
int *proj_warpdispl_d;
int *back_buffdispl_d;
int *back_mapdispl_d;
int *back_mapnz_d;
int *back_buffmap_d;
int *back_warpdispl_d;
#ifdef MATRIX
matrix *proj_warpindval_d;
matrix *back_warpindval_d;
#else
unsigned short *proj_warpindex_d;
MATPREC *proj_warpvalue_d;
unsigned short *back_warpindex_d;
MATPREC *back_warpvalue_d;
#endif

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
extern MPI_Comm MPI_COMM_BATCH;
extern int numproc_batch;
extern int myid_batch;
extern MPI_Comm MPI_COMM_DATA;
extern int numproc_data;
extern int myid_data;
extern MPI_Comm MPI_COMM_NODE;
extern int numproc_node;
extern int myid_node;
extern int numnode;
extern MPI_Comm MPI_COMM_SOCKET;
extern int numproc_socket;
extern int myid_socket;
extern int numsocket;

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

VECPREC *tomobuff_d;
VECPREC *partbuff_d;
COMMPREC *socketreducesendbuff_d;
COMMPREC *socketreducerecvbuff_d;
COMMPREC *nodereducesendbuff_d;
COMMPREC *nodereducerecvbuff_d;
COMMPREC *nodesendbuff_d;
COMMPREC *noderecvbuff_d;
COMMPREC *nodesendbuff_h;
COMMPREC *noderecvbuff_h;

extern int *socketrecvbuffdispl_p;
extern COMMPREC **socketrecvbuff_p;
extern int *socketrecvdevice_p;
extern int *noderecvbuffdispl_p;
extern COMMPREC **noderecvbuff_p;
extern int *noderecvdevice_p;

#ifdef MATRIX
__global__ void kernel_project __launch_bounds__(1024,1) (VECPREC *y, VECPREC *x, matrix *indval, int numrow, int numcol, int *buffdispl, int *displ, int *mapdispl, int *mapnz, int *buffmap, int buffsize){
#else
__global__ void kernel_project __launch_bounds__(1024,1) (VECPREC *y, VECPREC *x, unsigned short *index, MATPREC *value, int numrow, int numcol, int *buffdispl, int *displ, int *mapdispl, int *mapnz, int *buffmap, int buffsize){
#endif
  extern __shared__ VECPREC shared[];
  #ifdef MIXED
  float acc[FFACTOR] = {0.0};
  #else
  VECPREC acc[FFACTOR] = {0.0};
  #endif
  int wind = threadIdx.x%WARPSIZE;
  for(int buff = buffdispl[blockIdx.x]; buff < buffdispl[blockIdx.x+1]; buff++){
    int mapoffset = mapdispl[buff];
    for(int i = threadIdx.x; i < mapnz[buff]; i += blockDim.x){
      int ind = buffmap[mapoffset+i];
      #pragma unroll
      for(int f = 0; f < FFACTOR; f++)
        shared[f*buffsize+i] = x[f*numcol+ind];
    }
    __syncthreads();
    int warp = (buff*blockDim.x+threadIdx.x)/WARPSIZE;
    for(int n = displ[warp]; n < displ[warp+1]; n++){
      #ifdef MATRIX
      matrix mat = indval[n*(long)WARPSIZE+wind];
        #ifdef MIXED
      float val = mat.val;
      #pragma unroll
      for(int f = 0; f < FFACTOR; f++)
        acc[f] += __half2float(shared[f*buffsize+mat.ind])*val;
        #else
      for(int f = 0; f < FFACTOR; f++)
        acc[f] += shared[f*buffsize+mat.ind]*mat.val;
        #endif
      #else
      unsigned short ind = index[n*(long)WARPSIZE+wind];
      MATPREC val = value[n*(long)WARPSIZE+wind];
      #pragma unroll
      for(int f = 0; f < FFACTOR; f++)
        acc[f] += shared[f*buffsize+ind]*val;
      #endif
    }
    __syncthreads();
  }
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  if(row  < numrow)
    for(int f = 0; f < FFACTOR; f++)
      y[f*numrow+row] = acc[f];
};
__global__ void kernel_reduce(COMMPREC*,COMMPREC*,int*,int*,int,int,int*,int*);
__global__ void kernel_reducenopack(double*,COMMPREC*,int*,int*,int,int,int*,double);
__global__ void kernel_scatternopack(double*,COMMPREC*,int*,int*,int,int,int*,double);
__global__ void kernel_scatter(COMMPREC*,COMMPREC*,int*,int*,int,int,int*,int*);
__global__ void kernel_double2VECPREC(VECPREC*,double*,int,double);
__global__ void kernel_VECPREC2double(double*,VECPREC*,int,double);
__global__ void kernel_VECPREC2COMMPREC(COMMPREC*,VECPREC*,int,int*);
__global__ void kernel_COMMPREC2VECPREC(VECPREC*,COMMPREC*,int,int*);

void partial_project();
void partial_backproject();
double *reducebuff_d;
double *reducebuff_h;

int numdevice;
int mydevice;
cudaEvent_t start,stop;
float milliseconds;
MPI_Request *sendrequest;
MPI_Request *recvrequest;
cudaStream_t *socketstream;
cudaStream_t *nodestream;

void setup_gpu(double **obj_d, double **gra_d, double **dir_d, double **res_d, double **ray_d, double **obj_h, double **res_h){

  cudaGetDeviceCount(&numdevice);
  mydevice = myid%numdevice;
  cudaSetDevice(mydevice);
  if(myid==0){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("\n");
    printf("Device Count: %d\n",deviceCount);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,0);
    printf("Device %d name: %s\n",0,deviceProp.name);
    printf("Clock Frequency: %f GHz\n",deviceProp.clockRate/1.e9);
    printf("Computational Capabilities: %d, %d\n",deviceProp.major,deviceProp.minor);
    printf("Maximum global memory size: %lu\n",deviceProp.totalGlobalMem);
    printf("Maximum constant memory size: %lu\n",deviceProp.totalConstMem);
    printf("Maximum shared memory size per block: %lu\n",deviceProp.sharedMemPerBlock);
    printf("Maximum block dimensions: %dx%dx%d\n",deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
    printf("Maximum grid dimensions: %dx%dx%d\n",deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
    printf("Maximum threads per block: %d\n",deviceProp.maxThreadsPerBlock);
    printf("Warp size: %d\n",deviceProp.warpSize);
    printf("32-bit Reg. per block: %d\n",deviceProp.regsPerBlock);
    printf("\n");
  }

  //CONJUGATE-GRADIENT BUFFERS
  double batchmem = 0.0;
  batchmem += sizeof(double)*mynumpix*batchsize/1.0e9;
  if(mynumpix > mynumray)
    batchmem += sizeof(double)*mynumpix*batchsize/1.0e9;
  else
    batchmem += sizeof(double)*mynumray*batchsize/1.0e9;
  batchmem += sizeof(double)*mynumpix*batchsize/1.0e9;
  batchmem += sizeof(double)*mynumray*batchsize/1.0e9;
  cudaMalloc((void**)obj_d,sizeof(double)*mynumpix*batchsize);
  if(mynumpix > mynumray)
    cudaMalloc((void**)gra_d,sizeof(double)*mynumpix*batchsize);
  else
    cudaMalloc((void**)gra_d,sizeof(double)*mynumray*batchsize);
  cudaMalloc((void**)dir_d,sizeof(double)*mynumpix*batchsize);
  cudaMalloc((void**)res_d,sizeof(double)*mynumray*batchsize);
  *ray_d = *gra_d;
  cudaMallocHost((void**)obj_h,sizeof(double)*mynumpix*batchsize);
  cudaMallocHost((void**)res_h,sizeof(double)*mynumray*batchsize);
  //REDUCTION BUFFERS
  int reducebuffsize = 0;
  if(mynumpix > mynumray)
    reducebuffsize = (mynumpix*batchsize+255)/256;
  else
    reducebuffsize = (mynumray*batchsize+255)/256;
  if(myid==0)printf("reducebuffsize: %d\n",reducebuffsize);
  cudaMalloc((void**)&reducebuff_d,sizeof(double)*reducebuffsize);
  cudaMallocHost((void**)&reducebuff_h,sizeof(double)*reducebuffsize);

  double projmem = 0.0;
  projmem = projmem + sizeof(int)/1.0e9*(proj_numblocks+1);
  projmem = projmem + sizeof(int)/1.0e9*(proj_numbufftot+1);
  projmem = projmem + sizeof(int)/1.0e9*proj_numbufftot;
  projmem = projmem + sizeof(int)/1.0e9*proj_mapnztot;
  projmem = projmem + sizeof(int)/1.0e9*(proj_numbufftot*(proj_blocksize/WARPSIZE)+1);
  projmem = projmem + sizeof(unsigned short)/1.0e9*(proj_warpnztot*(long)WARPSIZE);
  projmem = projmem + sizeof(MATPREC)/1.0e9*(proj_warpnztot*(long)WARPSIZE);
  projmem = projmem + sizeof(int)/1.0e9*proj_mapnztot;
  //printf("PROC %d FORWARD PROJECTION MEMORY: %f GB\n",myid,projmem);
  double backmem = 0.0;
  backmem = backmem + sizeof(int)/1.0e9*(back_numblocks+1);
  backmem = backmem + sizeof(int)/1.0e9*(back_numbufftot+1);
  backmem = backmem + sizeof(int)/1.0e9*back_numbufftot;
  backmem = backmem + sizeof(int)/1.0e9*back_mapnztot;
  backmem = backmem + sizeof(int)/1.0e9*(back_numbufftot*(back_blocksize/WARPSIZE)+1);
  backmem = backmem + sizeof(unsigned short)/1.0e9*(back_warpnztot*(long)WARPSIZE);
  backmem = backmem + sizeof(MATPREC)/1.0e9*(back_warpnztot*(long)WARPSIZE);
  backmem = backmem + sizeof(int)/1.0e9*back_mapnztot;
  //printf("PROC %d BACKPROJECTION MEMORY: %f GB\n",myid,backmem);

  cudaMalloc((void**)&proj_buffdispl_d,sizeof(int)*(proj_numblocks+1));
  cudaMalloc((void**)&proj_mapdispl_d,sizeof(int)*(proj_numbufftot+1));
  cudaMalloc((void**)&proj_mapnz_d,sizeof(int)*proj_numbufftot);
  cudaMalloc((void**)&proj_buffmap_d,sizeof(int)*proj_mapnztot);
  cudaMalloc((void**)&proj_warpdispl_d,sizeof(int)*(proj_numbufftot*(proj_blocksize/WARPSIZE)+1));
  cudaMemcpy(proj_buffdispl_d,proj_buffdispl,sizeof(int)*(proj_numblocks+1),cudaMemcpyHostToDevice);
  cudaMemcpy(proj_mapdispl_d,proj_mapdispl,sizeof(int)*(proj_numbufftot+1),cudaMemcpyHostToDevice);
  cudaMemcpy(proj_mapnz_d,proj_mapnz,sizeof(int)*proj_numbufftot,cudaMemcpyHostToDevice);
  cudaMemcpy(proj_buffmap_d,proj_buffmap,sizeof(int)*proj_mapnztot,cudaMemcpyHostToDevice);
  cudaMemcpy(proj_warpdispl_d,proj_warpdispl,sizeof(int)*(proj_numbufftot*(proj_blocksize/WARPSIZE)+1),cudaMemcpyHostToDevice);
  delete[] proj_buffdispl;
  delete[] proj_mapdispl;
  delete[] proj_mapnz;
  delete[] proj_buffmap;
  delete[] proj_warpdispl;
  cudaMalloc((void**)&back_buffdispl_d,sizeof(int)*(back_numblocks+1));
  cudaMalloc((void**)&back_mapdispl_d,sizeof(int)*(back_numbufftot+1));
  cudaMalloc((void**)&back_mapnz_d,sizeof(int)*back_numbufftot);
  cudaMalloc((void**)&back_buffmap_d,sizeof(int)*back_mapnztot);
  cudaMalloc((void**)&back_warpdispl_d,sizeof(int)*(back_numbufftot*(back_blocksize/WARPSIZE)+1));
  cudaMemcpy(back_buffdispl_d,back_buffdispl,sizeof(int)*(back_numblocks+1),cudaMemcpyHostToDevice);
  cudaMemcpy(back_mapdispl_d,back_mapdispl,sizeof(int)*(back_numbufftot+1),cudaMemcpyHostToDevice);
  cudaMemcpy(back_mapnz_d,back_mapnz,sizeof(int)*back_numbufftot,cudaMemcpyHostToDevice);
  cudaMemcpy(back_buffmap_d,back_buffmap,sizeof(int)*back_mapnztot,cudaMemcpyHostToDevice);
  cudaMemcpy(back_warpdispl_d,back_warpdispl,sizeof(int)*(back_numbufftot*(back_blocksize/WARPSIZE)+1),cudaMemcpyHostToDevice);
  delete[] back_buffdispl;
  delete[] back_mapdispl;
  delete[] back_mapnz;
  delete[] back_buffmap;
  delete[] back_warpdispl;
  #ifdef MATRIX
  cudaMalloc((void**)&proj_warpindval_d,sizeof(matrix)*proj_warpnztot*(long)WARPSIZE);
  cudaMalloc((void**)&back_warpindval_d,sizeof(matrix)*back_warpnztot*(long)WARPSIZE);
  cudaMemcpy(proj_warpindval_d,proj_warpindval,sizeof(matrix)*proj_warpnztot*(long)WARPSIZE,cudaMemcpyHostToDevice);
  cudaMemcpy(back_warpindval_d,back_warpindval,sizeof(matrix)*back_warpnztot*(long)WARPSIZE,cudaMemcpyHostToDevice);
  delete[] proj_warpindval;
  delete[] back_warpindval;
  #else
  cudaMalloc((void**)&proj_warpindex_d,sizeof(unsigned short)*proj_warpnztot*(long)WARPSIZE);
  cudaMalloc((void**)&proj_warpvalue_d,sizeof(MATPREC)*proj_warpnztot*(long)WARPSIZE);
  cudaMalloc((void**)&back_warpindex_d,sizeof(unsigned short)*back_warpnztot*(long)WARPSIZE);
  cudaMalloc((void**)&back_warpvalue_d,sizeof(MATPREC)*back_warpnztot*(long)WARPSIZE);
  cudaMemcpy(proj_warpindex_d,proj_warpindex,sizeof(unsigned short)*proj_warpnztot*(long)WARPSIZE,cudaMemcpyHostToDevice);
  cudaMemcpy(proj_warpvalue_d,proj_warpvalue,sizeof(MATPREC)*proj_warpnztot*(long)WARPSIZE,cudaMemcpyHostToDevice);
  cudaMemcpy(back_warpindex_d,back_warpindex,sizeof(unsigned short)*back_warpnztot*(long)WARPSIZE,cudaMemcpyHostToDevice);
  cudaMemcpy(back_warpvalue_d,back_warpvalue,sizeof(MATPREC)*back_warpnztot*(long)WARPSIZE,cudaMemcpyHostToDevice);
  delete[] proj_warpindex;
  delete[] proj_warpvalue;
  delete[] back_warpindex;
  delete[] back_warpvalue;
  #endif

  //COMMUNICATION BUFFERS
  double commem = 0.0;
  commem += sizeof(VECPREC)*mynumpix*FFACTOR/1.0e9;
  commem += sizeof(VECPREC)*raynumout*FFACTOR/1.0e9;
  commem += sizeof(COMMPREC)*socketsendcommdispl[numproc_socket]*FFACTOR/1.0e9;
  commem += sizeof(COMMPREC)*socketrecvcommdispl[numproc_socket]*FFACTOR/1.0e9;
  commem += sizeof(COMMPREC)*nodesendcommdispl[numproc_node]*FFACTOR/1.0e9;
  commem += sizeof(COMMPREC)*noderecvcommdispl[numproc_node]*FFACTOR/1.0e9;
  commem += sizeof(COMMPREC)*nodereduceoutdispl[numproc_data]*FFACTOR/1.0e9;
  commem += sizeof(COMMPREC)*nodereduceincdispl[numproc_data]*FFACTOR/1.0e9;
  cudaMalloc((void**)&tomobuff_d,sizeof(VECPREC)*mynumpix*FFACTOR);
  cudaMalloc((void**)&partbuff_d,sizeof(VECPREC)*raynumout*FFACTOR);
  cudaMalloc((void**)&socketreducesendbuff_d,sizeof(COMMPREC)*socketsendcommdispl[numproc_socket]*FFACTOR);
  cudaMalloc((void**)&socketreducerecvbuff_d,sizeof(COMMPREC)*socketrecvcommdispl[numproc_socket]*FFACTOR);
  cudaMalloc((void**)&nodereducesendbuff_d,sizeof(COMMPREC)*nodesendcommdispl[numproc_node]*FFACTOR);
  cudaMalloc((void**)&nodereducerecvbuff_d,sizeof(COMMPREC)*noderecvcommdispl[numproc_node]*FFACTOR);
  cudaMalloc((void**)&nodesendbuff_d,sizeof(COMMPREC)*nodereduceoutdispl[numproc_data]*FFACTOR);
  cudaMalloc((void**)&noderecvbuff_d,sizeof(COMMPREC)*nodereduceincdispl[numproc_data]*FFACTOR);
  //HOST BUFFER
  cudaMallocHost((void**)&nodesendbuff_h,sizeof(COMMPREC)*nodereduceoutdispl[numproc_data]*FFACTOR);
  cudaMallocHost((void**)&noderecvbuff_h,sizeof(COMMPREC)*nodereduceincdispl[numproc_data]*FFACTOR);
  //PACK AND UNPACK MAPS
  commem += sizeof(int)*socketsendcommdispl[numproc_socket]*FFACTOR/1.0e9;
  commem += sizeof(int)*socketrecvcommdispl[numproc_socket]*FFACTOR/1.0e9;
  commem += sizeof(int)*(socketreduceoutdispl[numproc_data]+1)/1.0e9;
  commem += sizeof(int)*socketreducedispl[socketreduceoutdispl[numproc_data]]/1.0e9;
  commem += sizeof(int)*socketreduceoutdispl[numproc_data]*FFACTOR/1.0e9;
  commem += sizeof(int)*noderecvcommdispl[numproc_node]*FFACTOR/1.0e9;
  commem += sizeof(int)*(nodereduceoutdispl[numproc_data]+1)/1.0e9;
  commem += sizeof(int)*nodereducedispl[nodereduceoutdispl[numproc_data]]/1.0e9;
  commem += sizeof(int)*nodereduceoutdispl[numproc_data]*FFACTOR/1.0e9;
  commem += sizeof(int)*nodereduceincdispl[numproc_data]*FFACTOR/1.0e9;
  commem += sizeof(int)*(mynumray+1)/1.0e9;
  commem += sizeof(int)*noderaydispl[mynumray]/1.0e9;
  cudaMalloc((void**)&socketpackmap_d,sizeof(int)*socketsendcommdispl[numproc_socket]*FFACTOR);
  cudaMalloc((void**)&socketunpackmap_d,sizeof(int)*socketrecvcommdispl[numproc_socket]*FFACTOR);
  cudaMalloc((void**)&socketreducedispl_d,sizeof(int)*(socketreduceoutdispl[numproc_data]+1));
  cudaMalloc((void**)&socketreduceindex_d,sizeof(int)*socketreducedispl[socketreduceoutdispl[numproc_data]]);
  cudaMalloc((void**)&nodepackmap_d,sizeof(int)*socketreduceoutdispl[numproc_data]*FFACTOR);
  cudaMalloc((void**)&nodeunpackmap_d,sizeof(int)*noderecvcommdispl[numproc_node]*FFACTOR);
  cudaMalloc((void**)&nodereducedispl_d,sizeof(int)*(nodereduceoutdispl[numproc_data]+1));
  cudaMalloc((void**)&nodereduceindex_d,sizeof(int)*nodereducedispl[nodereduceoutdispl[numproc_data]]);
  cudaMalloc((void**)&raypackmap_d,sizeof(int)*nodereduceoutdispl[numproc_data]*FFACTOR);
  cudaMalloc((void**)&rayunpackmap_d,sizeof(int)*nodereduceincdispl[numproc_data]*FFACTOR);
  cudaMalloc((void**)&noderaydispl_d,sizeof(int)*(mynumray+1));
  cudaMalloc((void**)&noderayindex_d,sizeof(int)*noderaydispl[mynumray]);
  cudaMemcpy(socketpackmap_d,socketpackmap,sizeof(int)*socketsendcommdispl[numproc_socket]*FFACTOR,cudaMemcpyHostToDevice);
  cudaMemcpy(socketunpackmap_d,socketunpackmap,sizeof(int)*socketrecvcommdispl[numproc_socket]*FFACTOR,cudaMemcpyHostToDevice);
  cudaMemcpy(socketreducedispl_d,socketreducedispl,sizeof(int)*(socketreduceoutdispl[numproc_data]+1),cudaMemcpyHostToDevice);
  cudaMemcpy(socketreduceindex_d,socketreduceindex,sizeof(int)*socketreducedispl[socketreduceoutdispl[numproc_data]],cudaMemcpyHostToDevice);
  cudaMemcpy(nodepackmap_d,nodepackmap,sizeof(int)*socketreduceoutdispl[numproc_data]*FFACTOR,cudaMemcpyHostToDevice);
  cudaMemcpy(nodeunpackmap_d,nodeunpackmap,sizeof(int)*noderecvcommdispl[numproc_node]*FFACTOR,cudaMemcpyHostToDevice);
  cudaMemcpy(nodereducedispl_d,nodereducedispl,sizeof(int)*(nodereduceoutdispl[numproc_data]+1),cudaMemcpyHostToDevice);
  cudaMemcpy(nodereduceindex_d,nodereduceindex,sizeof(int)*nodereducedispl[nodereduceoutdispl[numproc_data]],cudaMemcpyHostToDevice);
  cudaMemcpy(raypackmap_d,raypackmap,sizeof(int)*nodereduceoutdispl[numproc_data]*FFACTOR,cudaMemcpyHostToDevice);
  cudaMemcpy(rayunpackmap_d,rayunpackmap,sizeof(int)*nodereduceincdispl[numproc_data]*FFACTOR,cudaMemcpyHostToDevice);
  cudaMemcpy(noderaydispl_d,noderaydispl,sizeof(int)*(mynumray+1),cudaMemcpyHostToDevice);
  cudaMemcpy(noderayindex_d,noderayindex,sizeof(int)*noderaydispl[mynumray],cudaMemcpyHostToDevice);

  double gpumem = projmem+backmem;
  double gpumems[numproc_data];
  double batchmems[numproc_data];
  double commems[numproc_data];
  MPI_Allgather(&gpumem,1,MPI_DOUBLE,gpumems,1,MPI_DOUBLE,MPI_COMM_DATA);
  MPI_Allgather(&batchmem,1,MPI_DOUBLE,batchmems,1,MPI_DOUBLE,MPI_COMM_DATA);
  MPI_Allgather(&commem,1,MPI_DOUBLE,commems,1,MPI_DOUBLE,MPI_COMM_DATA);
  if(myid==0){
    double gpumaxmem = 0.0;
    double batchmaxmem = 0.0;
    double commaxmem = 0.0;
    double totmaxmem = 0.0;
    double gputotmem = 0.0;
    double batchtotmem = 0.0;
    double commtotmem = 0.0;
    for(int p = 0; p < numproc_data; p++){
      printf("PROC %d GPU MEMORY: %f GB + %f GB + %f GB = %f GB\n",p,gpumems[p],batchmems[p],commems[p],gpumems[p]+batchmems[p]+commems[p]);
      if(gpumems[p]>gpumaxmem)gpumaxmem=gpumems[p];
      if(batchmems[p]>batchmaxmem)batchmaxmem=batchmems[p];
      if(commems[p]>commaxmem)commaxmem=commems[p];
      if(gpumems[p]+batchmems[p]+commems[p]>totmaxmem)totmaxmem=gpumems[p]+batchmems[p]+commems[p];
      gputotmem += gpumems[p];
      batchtotmem += batchmems[p];
      commtotmem += commems[p];
    }
    printf("MAX GPU MEMORY gpumem %f GB batchmem %f GB commem %f GB total %f GB\n",gpumaxmem,batchmaxmem,commaxmem,totmaxmem);
    printf("TOTAL GPU MEMORY gpumem %f GB + batchmem %f GB + commem %f GB = %f GB\n",gputotmem,batchtotmem,commtotmem,gputotmem+batchtotmem+commtotmem);
  }

  cudaFuncSetAttribute(kernel_project,cudaFuncAttributeMaxDynamicSharedMemorySize,(164-1)*1024);
  cudaFuncSetAttribute(kernel_project,cudaFuncAttributePreferredSharedMemoryCarveout,cudaSharedmemCarveoutMaxShared);

  cudaFuncAttributes funcAttributes;
  cudaFuncGetAttributes(&funcAttributes,kernel_project);
  if(myid==0){
    printf("\n");
    printf("SpMM Attributes\n");
    printf("Binary Version: %d\n",funcAttributes.binaryVersion);
    printf("Cache Mode: %d\n",funcAttributes.cacheModeCA);
    printf("Constant Memory: %lu\n",funcAttributes.constSizeBytes);
    printf("Local Memory: %lu\n",funcAttributes.localSizeBytes);
    printf("Max Dynamic Shared Memory: %d\n",funcAttributes.maxDynamicSharedSizeBytes);
    printf("Max Threads per Block: %d\n",funcAttributes.maxThreadsPerBlock);
    printf("Number of Registers: %d\n",funcAttributes.numRegs);
    printf("Shared Memory Carveout: %d\n",funcAttributes.preferredShmemCarveout);
    printf("PTX Version %d\n",funcAttributes.ptxVersion);
    printf("Static Shared Memory: %lu\n",funcAttributes.sharedSizeBytes);
    printf("\n");
  }

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  sendrequest = new MPI_Request[numproc_data];
  recvrequest = new MPI_Request[numproc_data];
  socketstream = new cudaStream_t[numproc_socket];
  nodestream = new cudaStream_t[numproc_node];
  for(int p = 0; p < numproc_socket; p++)
    cudaStreamCreate(&socketstream[p]);
  for(int p = 0; p < numproc_node; p++)
    cudaStreamCreate(&nodestream[p]);

  communications();
  return;
}

void project(double *sino_d, double *tomo_d, double scale, int batchslice){
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_DATA);
  double projecttime = MPI_Wtime();
  //PARTIAL PROJECTION
  kernel_double2VECPREC<<<(mynumpix*FFACTOR+255)/256,256>>>(tomobuff_d,tomo_d,mynumpix*FFACTOR,scale);
  partial_project();
  for(int slice = 0; slice < batchslice; slice += FFACTOR){
    //MEMCPY DEVICE TO HOST
    cudaEventRecord(start);
    cudaMemcpy(nodesendbuff_h,nodesendbuff_d,sizeof(COMMPREC)*nodereduceoutdispl[numproc_data]*FFACTOR,cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    pmtime += milliseconds/1e3;
    //HOST COMMUNICATION
    MPI_Barrier(MPI_COMM_DATA);
    double chtime = MPI_Wtime();
    {
      int sendcount = 0;
      int recvcount = 0;
      for(int p = 0; p < numproc_data; p++)
        if(nodereduceout[p]){
          MPI_Issend(nodesendbuff_h+nodereduceoutdispl[p]*FFACTOR,nodereduceout[p]*FFACTOR*sizeof(COMMPREC),MPI_BYTE,p,0,MPI_COMM_DATA,sendrequest+sendcount);
	  sendcount++;
	}
      for(int p = 0; p < numproc_data; p++)
        if(nodereduceinc[p]){
          MPI_Irecv(noderecvbuff_h+nodereduceincdispl[p]*FFACTOR,nodereduceinc[p]*FFACTOR*sizeof(COMMPREC),MPI_BYTE,p,0,MPI_COMM_DATA,recvrequest+recvcount);
          recvcount++;
        }
      #ifdef OVERLAP
      //PARTIAL PROJECTION
      if(slice+FFACTOR < batchslice){
        kernel_double2VECPREC<<<(mynumpix*FFACTOR+255)/256,256>>>(tomobuff_d,tomo_d+(slice+FFACTOR)*mynumpix,mynumpix*FFACTOR,scale);
        partial_project();
      }
      #endif
      MPI_Waitall(sendcount,sendrequest,MPI_STATUSES_IGNORE);
      MPI_Waitall(recvcount,recvrequest,MPI_STATUSES_IGNORE);
    }
    MPI_Barrier(MPI_COMM_DATA);
    pchtime += MPI_Wtime()-chtime;
    //if(myid==0)printf("rack time %e\n",MPI_Wtime()-chtime); 
    //MEMCPY HOST TO DEVICE
    cudaEventRecord(start);
    cudaMemcpy(noderecvbuff_d,noderecvbuff_h,sizeof(COMMPREC)*nodereduceincdispl[numproc_data]*FFACTOR,cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    pmtime += milliseconds/1e3;
    //HOST REDUCTION
    cudaEventRecord(start);
    kernel_reducenopack<<<(mynumray+255)/256,256>>>(sino_d+slice*mynumray,noderecvbuff_d,noderaydispl_d,noderayindex_d,mynumray,nodereduceincdispl[numproc_data],rayunpackmap_d,1.0/scale);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    prtime += milliseconds/1e3;
    //#endif
    numproj++;
    #ifndef OVERLAP
    //PARTIAL PROJECTION
    if(slice+FFACTOR < batchslice){
      kernel_double2VECPREC<<<(mynumpix*FFACTOR+255)/256,256>>>(tomobuff_d,tomo_d+(slice+FFACTOR)*mynumpix,mynumpix*FFACTOR,scale);
      partial_project();
    }
    #endif
  }
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_DATA);
  ptime += MPI_Wtime()-projecttime;
}


void backproject(double *tomo_d, double *sino_d, double scale, int batchslice){
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_DATA);
  double backprojecttime = MPI_Wtime();
  //HOST SCATTER
  cudaEventRecord(start);
  kernel_scatternopack<<<(mynumray+255)/256,256>>>(sino_d,noderecvbuff_d,noderaydispl_d,noderayindex_d,mynumray,nodereduceincdispl[numproc_data],rayunpackmap_d,scale);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds,start,stop);
  brtime += milliseconds/1e3;
  //MEMCPY DEVICE TO HOST
  cudaEventRecord(start);
  cudaMemcpy(noderecvbuff_h,noderecvbuff_d,sizeof(COMMPREC)*nodereduceincdispl[numproc_data]*FFACTOR,cudaMemcpyDeviceToHost);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds,start,stop);
  bmtime += milliseconds/1e3;
  //HOST COMMUNICATION
  MPI_Barrier(MPI_COMM_DATA);
  double chtime = MPI_Wtime();
  {
    int sendcount = 0;
    int recvcount = 0;
    for(int p = 0; p < numproc_data; p++)
      if(nodereduceout[p]){
        MPI_Irecv(nodesendbuff_h+nodereduceoutdispl[p]*FFACTOR,nodereduceout[p]*FFACTOR*sizeof(COMMPREC),MPI_BYTE,p,0,MPI_COMM_DATA,sendrequest+sendcount);
        sendcount++;
      }
    for(int p = 0; p < numproc_data; p++)
      if(nodereduceinc[p]){
        MPI_Issend(noderecvbuff_h+nodereduceincdispl[p]*FFACTOR,nodereduceinc[p]*FFACTOR*sizeof(COMMPREC),MPI_BYTE,p,0,MPI_COMM_DATA,recvrequest+recvcount);
	recvcount++;
      }
    MPI_Waitall(sendcount,sendrequest,MPI_STATUSES_IGNORE);
    MPI_Waitall(recvcount,recvrequest,MPI_STATUSES_IGNORE);
  }
  MPI_Barrier(MPI_COMM_DATA);
  bchtime += MPI_Wtime()-chtime;
  //if(myid==0)printf("rack time %e\n",MPI_Wtime()-chtime);
  //MEMCPY HOST TO DEVICE
  cudaEventRecord(start);
  cudaMemcpy(nodesendbuff_d,nodesendbuff_h,sizeof(COMMPREC)*nodereduceoutdispl[numproc_data]*FFACTOR,cudaMemcpyHostToDevice);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds,start,stop);
  bmtime += milliseconds/1e3;
  for(int slice = 0; slice < batchslice; slice += FFACTOR){
    double chtime;
    int sendcount = 0;
    int recvcount = 0;
    if(slice+FFACTOR < batchslice){
      //HOST SCATTER
      cudaEventRecord(start);
      kernel_scatternopack<<<(mynumray+255)/256,256>>>(sino_d+(slice+FFACTOR)*mynumray,noderecvbuff_d,noderaydispl_d,noderayindex_d,mynumray,nodereduceincdispl[numproc_data],rayunpackmap_d,scale);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds,start,stop);
      brtime += milliseconds/1e3;
      //MEMCPY DEVICE TO HOST
      cudaEventRecord(start);
      cudaMemcpy(noderecvbuff_h,noderecvbuff_d,sizeof(COMMPREC)*nodereduceincdispl[numproc_data]*FFACTOR,cudaMemcpyDeviceToHost);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds,start,stop);
      bmtime += milliseconds/1e3;
      //HOST COMMUNICATION
      MPI_Barrier(MPI_COMM_DATA);
      chtime = MPI_Wtime();
      for(int p = 0; p < numproc_data; p++)
        if(nodereduceout[p]){
          MPI_Irecv(nodesendbuff_h+nodereduceoutdispl[p]*FFACTOR,nodereduceout[p]*FFACTOR*sizeof(COMMPREC),MPI_BYTE,p,0,MPI_COMM_DATA,sendrequest+sendcount);
          sendcount++;
        }
      for(int p = 0; p < numproc_data; p++)
        if(nodereduceinc[p]){
          MPI_Issend(noderecvbuff_h+nodereduceincdispl[p]*FFACTOR,nodereduceinc[p]*FFACTOR*sizeof(COMMPREC),MPI_BYTE,p,0,MPI_COMM_DATA,recvrequest+recvcount);
	  recvcount++;
	}
    }
    #ifdef OVERLAP
    //PARTIAL BACKPROJECTION
    partial_backproject();
    kernel_VECPREC2double<<<(mynumpix*FFACTOR+255)/256,256>>>(tomo_d+slice*mynumpix,tomobuff_d,mynumpix*FFACTOR,1.0/scale);
    #endif
    if(slice+FFACTOR < batchslice){
      MPI_Waitall(sendcount,sendrequest,MPI_STATUSES_IGNORE);
      MPI_Waitall(recvcount,recvrequest,MPI_STATUSES_IGNORE);
      MPI_Barrier(MPI_COMM_DATA);
      bchtime += MPI_Wtime()-chtime;
      //if(myid==0)printf("rack time %e\n",MPI_Wtime()-chtime);
    }
    #ifndef OVERLAP
    //PARTIAL BACKPROJECTION
    partial_backproject();
    kernel_VECPREC2double<<<(mynumpix*FFACTOR+255)/256,256>>>(tomo_d+slice*mynumpix,tomobuff_d,mynumpix*FFACTOR,1.0/scale);
    #endif
    numback++;
    if(slice+FFACTOR < batchslice){
      //MEMCPY HOST TO DEVICE
      cudaEventRecord(start);
      cudaMemcpy(nodesendbuff_d,nodesendbuff_h,sizeof(COMMPREC)*nodereduceoutdispl[numproc_data]*FFACTOR,cudaMemcpyHostToDevice);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds,start,stop);
      bmtime += milliseconds/1e3;
    }
  }
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_DATA);
  btime += MPI_Wtime()-backprojecttime;
}


void partial_project(){
    cudaEventRecord(start);
    #ifdef MATRIX
    kernel_project<<<proj_numblocks,proj_blocksize,sizeof(VECPREC)*proj_buffsize*FFACTOR>>>(partbuff_d,tomobuff_d,proj_warpindval_d,raynumout,mynumpix,proj_buffdispl_d,proj_warpdispl_d,proj_mapdispl_d,proj_mapnz_d,proj_buffmap_d,proj_buffsize);
    #else
    kernel_project<<<proj_numblocks,proj_blocksize,sizeof(VECPREC)*proj_buffsize*FFACTOR>>>(partbuff_d,tomobuff_d,proj_warpindex_d,proj_warpvalue_d,raynumout,mynumpix,proj_buffdispl_d,proj_warpdispl_d,proj_mapdispl_d,proj_mapnz_d,proj_buffmap_d,proj_buffsize);
    #endif
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    //if(myid==0)printf("project %e milliseconds\n",milliseconds);
    pktime += milliseconds/1e3;
    //COMMUNICATION BUFFER
    kernel_VECPREC2COMMPREC<<<(raynumout*FFACTOR+255)/256,256>>>(socketreducesendbuff_d,partbuff_d,raynumout*FFACTOR,socketpackmap_d);
    cudaDeviceSynchronize();
    //SOCKET COMMUNICATION
    MPI_Barrier(MPI_COMM_SOCKET);
    double cstime = MPI_Wtime();
    for(int psend = 0; psend < numproc_socket; psend++)
      if(socketsendcomm[psend])
        cudaMemcpyPeerAsync(socketrecvbuff_p[psend]+socketrecvbuffdispl_p[psend]*FFACTOR,socketrecvdevice_p[psend],socketreducesendbuff_d+socketsendcommdispl[psend]*FFACTOR,mydevice,sizeof(COMMPREC)*socketsendcomm[psend]*FFACTOR,socketstream[psend]);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_SOCKET);
    pcstime += MPI_Wtime()-cstime;
    //if(myid==0)printf("socket time %e\n",MPI_Wtime()-cstime);
    //SOCKET REDUCTION
    cudaEventRecord(start);
    kernel_reduce<<<(socketreduceoutdispl[numproc_data]+255)/256,256>>>(nodereducesendbuff_d,socketreducerecvbuff_d,socketreducedispl_d,socketreduceindex_d,socketreduceoutdispl[numproc_data],socketrecvcommdispl[numproc_socket],nodepackmap_d,socketunpackmap_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    prtime += milliseconds/1e3;
    //NODE COMMUNICATION
    MPI_Barrier(MPI_COMM_NODE);
    double cntime = MPI_Wtime();
    for(int psend = 0; psend < numproc_node; psend++)
      if(nodesendcomm[psend])
        cudaMemcpyPeerAsync(noderecvbuff_p[psend]+noderecvbuffdispl_p[psend]*FFACTOR,noderecvdevice_p[psend],nodereducesendbuff_d+nodesendcommdispl[psend]*FFACTOR,mydevice,sizeof(COMMPREC)*nodesendcomm[psend]*FFACTOR,nodestream[psend]);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_NODE);
    pcntime += MPI_Wtime()-cntime;
    //if(myid==0)printf("node time %e\n",MPI_Wtime()-cntime);
    //NODE REDUCTION
    cudaEventRecord(start);
    kernel_reduce<<<(nodereduceoutdispl[numproc_data]+255)/256,256>>>(nodesendbuff_d,nodereducerecvbuff_d,nodereducedispl_d,nodereduceindex_d,nodereduceoutdispl[numproc_data],noderecvcommdispl[numproc_node],raypackmap_d,nodeunpackmap_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    prtime += milliseconds/1e3;
};


void partial_backproject(){
    //NODE SCATTER
    cudaEventRecord(start);
    kernel_scatter<<<(nodereduceoutdispl[numproc_data]+255)/256,256>>>(nodesendbuff_d,nodereducerecvbuff_d,nodereducedispl_d,nodereduceindex_d,nodereduceoutdispl[numproc_data],noderecvcommdispl[numproc_node],raypackmap_d,nodeunpackmap_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    brtime += milliseconds/1e3;
    //NODE COMMUNICATION
    MPI_Barrier(MPI_COMM_NODE);
    double cntime = MPI_Wtime();
    for(int psend = 0; psend < numproc_node; psend++)
      if(nodesendcomm[psend])
        cudaMemcpyPeerAsync(nodereducesendbuff_d+nodesendcommdispl[psend]*FFACTOR,mydevice,noderecvbuff_p[psend]+noderecvbuffdispl_p[psend]*FFACTOR,noderecvdevice_p[psend],sizeof(COMMPREC)*nodesendcomm[psend]*FFACTOR,nodestream[psend]);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_NODE);
    bcntime += MPI_Wtime()-cntime;
    //if(myid==0)printf("node time %e\n",MPI_Wtime()-cntime);
    //SOCKET SCATTER
    cudaEventRecord(start);
    kernel_scatter<<<(socketreduceoutdispl[numproc_data]+255)/256,256>>>(nodereducesendbuff_d,socketreducerecvbuff_d,socketreducedispl_d,socketreduceindex_d,socketreduceoutdispl[numproc_data],socketrecvcommdispl[numproc_socket],nodepackmap_d,socketunpackmap_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    brtime += milliseconds/1e3;
    //SOCKET COMMUNICATION
    MPI_Barrier(MPI_COMM_SOCKET);
    double cstime = MPI_Wtime();
    for(int psend = 0; psend < numproc_socket; psend++)
      if(socketsendcomm[psend])
        cudaMemcpyPeerAsync(socketreducesendbuff_d+socketsendcommdispl[psend]*FFACTOR,mydevice,socketrecvbuff_p[psend]+socketrecvbuffdispl_p[psend]*FFACTOR,socketrecvdevice_p[psend],sizeof(COMMPREC)*socketsendcomm[psend]*FFACTOR,socketstream[psend]);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_SOCKET);
    bcstime += MPI_Wtime()-cstime;
    //if(myid==0)printf("socket time %e\n",MPI_Wtime()-cstime);
    //BACKPROJECTION
    kernel_COMMPREC2VECPREC<<<(raynumout*FFACTOR+255)/256,256>>>(partbuff_d,socketreducesendbuff_d,raynumout*FFACTOR,socketpackmap_d);
    cudaEventRecord(start);
    #ifdef MATRIX
    kernel_project<<<back_numblocks,back_blocksize,sizeof(VECPREC)*back_buffsize*FFACTOR>>>(tomobuff_d,partbuff_d,back_warpindval_d,mynumpix,raynumout,back_buffdispl_d,back_warpdispl_d,back_mapdispl_d,back_mapnz_d,back_buffmap_d,back_buffsize);
    #else
    kernel_project<<<back_numblocks,back_blocksize,sizeof(VECPREC)*back_buffsize*FFACTOR>>>(tomobuff_d,partbuff_d,back_warpindex_d,back_warpvalue_d,mynumpix,raynumout,back_buffdispl_d,back_warpdispl_d,back_mapdispl_d,back_mapnz_d,back_buffmap_d,back_buffsize);
    #endif
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds,start,stop);
    //if(myid==0)printf("backproject %e milliseconds\n",milliseconds);
    bktime += milliseconds/1e3;
};
__global__ void kernel_reduce(COMMPREC *y, COMMPREC *x, int *displ, int *index, int numrow, int numcol, int *packmap, int *unpackmap){
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  #ifdef MIXED
  float reduce[FFACTOR] = {0.0};
  #else
  VECPREC reduce[FFACTOR] = {0.0};
  #endif
  if(row < numrow){
    for(int n = displ[row]; n < displ[row+1]; n++){
      int ind = index[n];
      for(int f = 0; f < FFACTOR; f++)
        #ifdef MIXED
        reduce[f] += __half2float(x[unpackmap[f*numcol+ind]]);
        #else
        reduce[f] += x[unpackmap[f*numcol+ind]];
        #endif
    }
    for(int f = 0; f < FFACTOR; f++)
      y[packmap[f*numrow+row]] = reduce[f];
  }
};
__global__ void kernel_reducenopack(double *y, COMMPREC *x, int *displ, int *index, int numrow, int numcol, int *unpackmap, double scale){
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  #ifdef MIXED
  float reduce[FFACTOR] = {0.0};
  #else
  VECPREC reduce[FFACTOR] = {0.0};
  #endif
  if(row < numrow){
    for(int n = displ[row]; n < displ[row+1]; n++){
      int ind = index[n];
      for(int f = 0; f < FFACTOR; f++)
        #ifdef MIXED
        reduce[f] += __half2float(x[unpackmap[f*numcol+ind]]);
        #else
        reduce[f] += x[unpackmap[f*numcol+ind]];
        #endif
    }
    for(int f = 0; f < FFACTOR; f++)
      y[f*numrow+row] = (double)reduce[f]*scale;
  }
};
__global__ void kernel_scatternopack(double *y, COMMPREC *x, int *displ, int *index, int numrow, int numcol, int *unpackmap, double scale){
  int row = blockIdx.x*blockDim.x+threadIdx.x;
  VECPREC scatter[FFACTOR] = {0.0};
  if(row < numrow){
    for(int f = 0; f < FFACTOR; f++)
      scatter[f] = y[f*numrow+row]*scale;
    for(int n = displ[row]; n < displ[row+1]; n++){
      int ind = index[n];
      for(int f = 0; f < FFACTOR; f++)
        x[unpackmap[f*numcol+ind]] = scatter[f];
    }
  }
};
__global__ void kernel_scatter(COMMPREC *y, COMMPREC *x, int *displ, int *index, int numrow, int numcol, int *packmap, int *unpackmap){
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
};
__global__ void kernel_double2VECPREC(VECPREC *y, double *x,int dim, double scale){
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid < dim)
    y[tid] = x[tid]*scale;
};
__global__ void kernel_VECPREC2double(double *y, VECPREC *x,int dim, double scale){
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid < dim)
    y[tid] = (double)x[tid]*scale;
};
__global__ void kernel_VECPREC2COMMPREC(COMMPREC *y, VECPREC *x,int dim, int *packmap){
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid < dim)
    y[packmap[tid]] = x[tid];
};
__global__ void kernel_COMMPREC2VECPREC(VECPREC *y, COMMPREC *x,int dim, int *unpackmap){
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid < dim)
    y[tid] = x[unpackmap[tid]];
};
void copyD2D_kernel(double *a, double *b, int dim){
  cudaMemcpy(a,b,sizeof(double)*dim,cudaMemcpyDeviceToDevice);
};
void copyD2H_kernel(double *a, double *b, int dim){
  cudaMemcpy(a,b,sizeof(double)*dim,cudaMemcpyDeviceToHost);
};
void copyH2D_kernel(double *a, double *b, int dim){
  cudaMemcpy(a,b,sizeof(double)*dim,cudaMemcpyHostToDevice);
};
void init_kernel(double *a, int dim){
  cudaMemset(a,0,sizeof(double)*dim);
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
__global__ void kernel_dot(double *a, double *b, int dim, double *buffer){
  extern __shared__ double temp[];
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid < dim)
    temp[threadIdx.x] = a[tid]*b[tid];
  else
    temp[threadIdx.x] = 0;
  for(int stride = blockDim.x/2; stride > 0; stride>>=1){
    __syncthreads();
    if(threadIdx.x < stride)
      temp[threadIdx.x] += temp[threadIdx.x+stride];
  }
  if(threadIdx.x==0)
    buffer[blockIdx.x] = temp[0];
};
double dot_kernel(double *a, double *b, int dim){
  int numblocks = (dim+255)/256;
  kernel_dot<<<numblocks,256,sizeof(double)*256>>>(a,b,dim,reducebuff_d);
  cudaMemcpy(reducebuff_h,reducebuff_d,sizeof(double)*numblocks,cudaMemcpyDeviceToHost);
  double reduce = 0.0;
  for(int n = 0; n < numblocks; n++)
    reduce += reducebuff_h[n];
  MPI_Allreduce(MPI_IN_PLACE,&reduce,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_DATA);
  return reduce;
};
__global__ void kernel_max(double *a, int dim, double *buffer){
  extern __shared__ double temp[];
  int tid = blockIdx.x*blockDim.x+threadIdx.x;
  if(tid < dim)
    temp[threadIdx.x] = a[tid];
  else
    temp[threadIdx.x] = 0.0;
  for(int stride = blockDim.x/2; stride > 0; stride>>=1){
    __syncthreads();
    if(threadIdx.x < stride)
      if(temp[threadIdx.x+stride] > temp[threadIdx.x])
        temp[threadIdx.x] = temp[threadIdx.x+stride];
  }
  if(threadIdx.x==0)
    buffer[blockIdx.x] = temp[0];
};
double max_kernel(double *a, int dim){
  int numblocks = (dim+255)/256;
  kernel_max<<<numblocks,256,sizeof(double)*256>>>(a,dim,reducebuff_d);
  cudaMemcpy(reducebuff_h,reducebuff_d,sizeof(double)*numblocks,cudaMemcpyDeviceToHost);
  double reduce = 0.0;
  for(int n = 0; n < numblocks; n++)
    if(reducebuff_h[n] > reduce)
      reduce = reducebuff_h[n];
  MPI_Allreduce(MPI_IN_PLACE,&reduce,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_DATA);
  return reduce;
};

#include "vars.h"

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

extern int *socketpackmap_d;
extern int *socketunpackmap_d;
extern int *socketreducedispl_d;
extern int *socketreduceindex_d;
extern int *nodepackmap_d;
extern int *nodeunpackmap_d;
extern int *nodereducedispl_d;
extern int *nodereduceindex_d;
extern int *raypackmap_d;
extern int *rayunpackmap_d;
extern int *noderaydispl_d;
extern int *noderayindex_d;

extern COMMPREC *socketreducesendbuff_d;
extern COMMPREC *socketreducerecvbuff_d;
extern COMMPREC *nodereducesendbuff_d;
extern COMMPREC *nodereducerecvbuff_d;
extern COMMPREC *nodesendbuff_d;
extern COMMPREC *noderecvbuff_d;
extern COMMPREC *nodesendbuff_h;
extern COMMPREC *noderecvbuff_h;

extern int numdevice;
extern int mydevice;

int *socketrecvbuffdispl_p;
COMMPREC **socketrecvbuff_p;
int *socketrecvdevice_p;
int *noderecvbuffdispl_p;
COMMPREC **noderecvbuff_p;
int *noderecvdevice_p;

void communications(){

  MPI_Request sendrequest[numproc];
  MPI_Request recvrequest[numproc];

  socketrecvbuff_p = new COMMPREC*[numproc_socket];
  socketrecvbuffdispl_p = new int[numproc_socket];
  cudaIpcMemHandle_t sockethandle[numproc_socket];
  //RECEIVER SENDS MEMORY HANDLE
  for(int precv = 0; precv < numproc_socket; precv++)
    if(socketrecvcomm[precv]){
      MPI_Isend(&socketrecvcommdispl[precv],1,MPI_INT,precv,1,MPI_COMM_SOCKET,sendrequest+precv);
      if(myid_socket!=precv){
        cudaIpcGetMemHandle(&sockethandle[precv],socketreducerecvbuff_d);
        MPI_Isend(&sockethandle[precv],sizeof(cudaIpcMemHandle_t),MPI_BYTE,precv,0,MPI_COMM_SOCKET,recvrequest+precv);
      }
      else
        socketrecvbuff_p[precv] = socketreducerecvbuff_d;
    }
  //SENDER OPENS MEMORY HANDLE
  for(int psend = 0; psend < numproc_socket; psend++)
    if(socketsendcomm[psend]){
      MPI_Recv(&socketrecvbuffdispl_p[psend],1,MPI_INT,psend,1,MPI_COMM_SOCKET,MPI_STATUS_IGNORE);
      if(myid_socket!=psend){
        cudaIpcMemHandle_t temphandle;
        MPI_Recv(&temphandle,sizeof(cudaIpcMemHandle_t),MPI_BYTE,psend,0,MPI_COMM_SOCKET,MPI_STATUS_IGNORE);
        cudaIpcOpenMemHandle((void**)&socketrecvbuff_p[psend],temphandle,cudaIpcMemLazyEnablePeerAccess);
      }
    }
  //RECEIVER DEVICE ID
  socketrecvdevice_p = new int[numproc_socket];
  for(int p = 0; p < numproc_socket; p++)
    socketrecvdevice_p[p] = ((myid/numproc_socket)*numproc_socket+p)%numdevice;
  //SOCKET IPC WARM-UP
  {
    MPI_Barrier(MPI_COMM_SOCKET);
    double time = MPI_Wtime();
    for(int psend = 0; psend < numproc_socket; psend++){
      if(socketsendcomm[psend])
        cudaMemcpyPeerAsync(socketrecvbuff_p[psend]+socketrecvbuffdispl_p[psend]*FFACTOR,socketrecvdevice_p[psend],socketreducesendbuff_d+socketsendcommdispl[psend]*FFACTOR,mydevice,sizeof(COMMPREC)*socketsendcomm[psend]*FFACTOR);
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_SOCKET);
    if(myid==0)printf("socket warmup time %e\n",MPI_Wtime()-time);
  }
  noderecvbuff_p = new COMMPREC*[numproc_node];
  noderecvbuffdispl_p = new int[numproc_node];
  cudaIpcMemHandle_t nodehandle[numproc_node];
  //RECEIVER SENDS MEMORY HANDLE
  for(int precv = 0; precv < numproc_node; precv++)
    if(noderecvcomm[precv]){
      MPI_Isend(&noderecvcommdispl[precv],1,MPI_INT,precv,1,MPI_COMM_NODE,sendrequest+precv);
      if(myid_node!=precv){
        cudaIpcGetMemHandle(&nodehandle[precv],nodereducerecvbuff_d);
        MPI_Isend(&nodehandle[precv],sizeof(cudaIpcMemHandle_t),MPI_BYTE,precv,0,MPI_COMM_NODE,recvrequest+precv);
      }
      else
        noderecvbuff_p[precv] = nodereducerecvbuff_d;
    }
  //RECEIVER SENDS MEMORY HANDLE
  for(int precv = 0; precv < numproc_node; precv++)
    if(noderecvcomm[precv]){
      MPI_Isend(&noderecvcommdispl[precv],1,MPI_INT,precv,1,MPI_COMM_NODE,sendrequest+precv);
      if(myid_node!=precv){
        cudaIpcGetMemHandle(&nodehandle[precv],nodereducerecvbuff_d);
        MPI_Isend(&nodehandle[precv],sizeof(cudaIpcMemHandle_t),MPI_BYTE,precv,0,MPI_COMM_NODE,recvrequest+precv);
      }
      else
        noderecvbuff_p[precv] = nodereducerecvbuff_d;
    }
  //SENDER OPENS MEMORY HANDLE
  for(int psend = 0; psend < numproc_node; psend++)
    if(nodesendcomm[psend]){
      MPI_Recv(&noderecvbuffdispl_p[psend],1,MPI_INT,psend,1,MPI_COMM_NODE,MPI_STATUS_IGNORE);
      if(myid_node!=psend){
        cudaIpcMemHandle_t temphandle;
        MPI_Recv(&temphandle,sizeof(cudaIpcMemHandle_t),MPI_BYTE,psend,0,MPI_COMM_NODE,MPI_STATUS_IGNORE);
        cudaIpcOpenMemHandle((void**)&noderecvbuff_p[psend],temphandle,cudaIpcMemLazyEnablePeerAccess);
      }
    }
  //RECEIVER DEVICE ID
  noderecvdevice_p = new int[numproc_node];
  for(int p = 0; p < numproc_node; p++)
    noderecvdevice_p[p] = ((myid/numproc_node)*numproc_node+p)%numdevice;
  //NODE IPC WARM-UP
  {
    MPI_Barrier(MPI_COMM_NODE);
    double time = MPI_Wtime();
    for(int psend = 0; psend < numproc_node; psend++){
      if(nodesendcomm[psend])
        cudaMemcpyPeerAsync(noderecvbuff_p[psend]+noderecvbuffdispl_p[psend]*FFACTOR,noderecvdevice_p[psend],nodereducesendbuff_d+nodesendcommdispl[psend]*FFACTOR,mydevice,sizeof(COMMPREC)*nodesendcomm[psend]*FFACTOR);
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_NODE);
    if(myid==0)printf("node warmup time %e\n",MPI_Wtime()-time);
  }
  //HOST IPC WARM-UP
  {
    MPI_Barrier(MPI_COMM_WORLD);
    double chtime = MPI_Wtime();
    {
      int sendcount = 0;
      int recvcount = 0;
      for(int p = 0; p < numproc; p++){
        if(nodereduceout[p]){
          MPI_Isend(nodesendbuff_h+nodereduceoutdispl[p]*FFACTOR,nodereduceout[p]*FFACTOR*sizeof(COMMPREC),MPI_BYTE,p,0,MPI_COMM_WORLD,sendrequest+sendcount);
          sendcount++;
        }
        if(nodereduceinc[p]){
          MPI_Irecv(noderecvbuff_h+nodereduceincdispl[p]*FFACTOR,nodereduceinc[p]*FFACTOR*sizeof(COMMPREC),MPI_BYTE,p,0,MPI_COMM_WORLD,recvrequest+recvcount);
          recvcount++;
        }
      }
      MPI_Waitall(recvcount,recvrequest,MPI_STATUSES_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(myid==0)printf("rack time %e\n",MPI_Wtime()-chtime);
  }
}

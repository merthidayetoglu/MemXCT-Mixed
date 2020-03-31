#include "vars.h"

extern int mynumray;
extern int raynuminc;
extern int raynumout;
extern long raynumoutall;

extern int *raysendstart;
extern int *rayrecvstart;
extern int *raysendcount;
extern int *rayrecvcount;

extern int *rayraystart;
extern int *rayrayind;

long socketrayoutall;
int *socketreduceout;
int *socketreduceinc;
int *socketreduceoutdispl;
int *socketreduceincdispl;
int *socketsendcomm;
int *socketrecvcomm;
int *socketsendcommdispl;
int *socketrecvcommdispl;
int *socketsendmap;
int *socketreducedispl;
int *socketreduceindex;
int *socketraydispl;
int *socketrayindex;
int *socketpackmap;
int *socketunpackmap;

long noderayoutall;
int *nodereduceout;
int *nodereduceinc;
int *nodereduceoutdispl;
int *nodereduceincdispl;
int *nodesendcomm;
int *noderecvcomm;
int *nodesendcommdispl;
int *noderecvcommdispl;
int *nodesendmap;
int *nodereducedispl;
int *nodereduceindex;
int *noderaydispl;
int *noderayindex;
int *nodepackmap;
int *nodeunpackmap;

int *raypackmap;
int *rayunpackmap;

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

void reducemap(){
  //SOCKET REDUCTION MAPS
  {
    int *socketmap = new int[raynuminc];
    int socketrayinc[numsocket];
    #pragma omp parallel for
    for(int socket = 0; socket < numsocket; socket++){
      for(int p = socket*numproc_socket; p < (socket+1)*numproc_socket; p++)
        for(int n = rayrecvstart[p]; n < rayrecvstart[p]+rayrecvcount[p]; n++)
          socketmap[n] = socket;
      socketrayinc[socket] = 0;
    }
    for(int m = 0; m < mynumray; m++){
      int count[numsocket];
      for(int socket = 0; socket < numsocket; socket++)
        count[socket] = 0;
      for(int n = rayraystart[m]; n < rayraystart[m+1]; n++)
        count[socketmap[rayrayind[n]]]++;
      for(int socket = 0; socket < numsocket; socket++)
        if(count[socket])
          socketrayinc[socket]++;
    }
    int socketrayincdispl[numsocket+1];
    int *socketnz[numsocket];
    socketrayincdispl[0] = 0;
    for(int socket = 0; socket < numsocket; socket++){
      socketrayincdispl[socket+1] = socketrayincdispl[socket] + socketrayinc[socket];
      socketnz[socket] = new int[socketrayinc[socket]];
      socketrayinc[socket] = 0;
    }
    socketrayoutall = socketrayincdispl[numsocket];
    MPI_Allreduce(MPI_IN_PLACE,&socketrayoutall,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    if(myid==0)printf("ALL COMMUNICATION: %ld (%f MB)\n",raynumoutall,raynumoutall*sizeof(COMMPREC)/1.0e6);
    if(myid==0)printf("INTER-SOCKET COMM: %ld (%f MB) %f%% saving\n",socketrayoutall,socketrayoutall*sizeof(COMMPREC)/1.0e6,(raynumoutall-socketrayoutall)/(double)raynumoutall*100);
    for(int m = 0; m < mynumray; m++){
      int count[numsocket];
      for(int socket = 0; socket < numsocket; socket++)
        count[socket] = 0;
      for(int n = rayraystart[m]; n < rayraystart[m+1]; n++)
        count[socketmap[rayrayind[n]]]++;
      for(int socket = 0; socket < numsocket; socket++)
        if(count[socket]){
          socketnz[socket][socketrayinc[socket]] = count[socket];
          socketrayinc[socket]++;
        }
    }
    int *socketdispl[numsocket];
    int *socketindex[numsocket];
    #pragma omp parallel for
    for(int socket = 0; socket < numsocket; socket++){
      socketdispl[socket] = new int[socketrayinc[socket]+1];
      socketdispl[socket][0] = 0;
      for(int k = 1; k < socketrayinc[socket]+1; k++)
        socketdispl[socket][k] = socketdispl[socket][k-1] + socketnz[socket][k-1];
      socketindex[socket] = new int[socketdispl[socket][socketrayinc[socket]]];
      socketrayinc[socket] = 0;
    }
    for(int m = 0; m < mynumray; m++){
      int count[numsocket];
      for(int socket = 0; socket < numsocket; socket++)
        count[socket] = 0;
      for(int n = rayraystart[m]; n < rayraystart[m+1]; n++){
        int index = rayrayind[n];
        int socket = socketmap[index];
        socketindex[socket][socketdispl[socket][socketrayinc[socket]]+count[socket]] = index;
        count[socket]++;
      }
      for(int socket = 0; socket < numsocket; socket++)
        if(count[socket])
          socketrayinc[socket]++;
    }
    delete[] socketmap;
    //FIND SOCKET REDUCTION MAPPING RECEIVING SIDE
    int *socketraymap = new int[raynuminc];
    #pragma omp parallel for
    for(int m = 0; m < mynumray; m++)
      for(int n = rayraystart[m]; n < rayraystart[m+1]; n++)
        socketraymap[rayrayind[n]] = m;
    int *socketinvmap = new int[socketrayincdispl[numsocket]];
    #pragma omp parallel for
    for(int socket = 0; socket < numsocket; socket++)
      for(int m = 0; m < socketrayinc[socket]; m++)
        for(int n = socketdispl[socket][m]; n < socketdispl[socket][m+1]; n++)
          socketinvmap[socketrayincdispl[socket]+m] = socketraymap[socketindex[socket][n]];
    delete[] socketraymap;
    int *socketraynz = new int[mynumray];
    #pragma omp parallel for
    for(int n = 0; n < mynumray; n++)
      socketraynz[n] = 0;
    for(int n = 0; n < socketrayincdispl[numsocket]; n++)
      socketraynz[socketinvmap[n]]++;
    socketraydispl = new int[mynumray+1];
    socketraydispl[0] = 0;
    for(int m = 1; m < mynumray+1; m++){
      socketraydispl[m] = socketraydispl[m-1] + socketraynz[m-1];
      socketraynz[m-1] = 0;
    }
    socketrayindex = new int[socketraydispl[mynumray]];
    for(int n = 0; n < socketrayincdispl[numsocket]; n++){
      int ray = socketinvmap[n];
      socketrayindex[socketraydispl[ray]+socketraynz[ray]] = n;
      socketraynz[ray]++;
    }
    delete[] socketinvmap;
    delete[] socketraynz;
    //FIND SOCKET REDUCTION MAPPING SENDING SIDE
    socketreduceout = new int[numproc];
    #pragma omp parallel for
    for(int p = 0; p < numproc; p++)
      socketreduceout[p] = 0;
    #pragma omp parallel for
    for(int socket = 0; socket < numsocket; socket++)
      socketreduceout[socket*numproc_socket+myid_socket] = socketrayinc[socket];
    /*#pragma omp parallel for
    for(int socket = 0; socket < numsocket; socket++){
      int socketouttemp = (socketrayinc[socket]/numproc_socket)*numproc_socket;
      for(int p = socket*numproc_socket; p < (socket+1)*numproc_socket; p++){
        socketreduceout[p] = socketrayinc[socket]/numproc_socket;
        if(socketouttemp < socketrayinc[socket]){
          socketreduceout[p]++;
          socketouttemp++;
        }
      }
    }*/
    socketreduceoutdispl = new int[numproc+1];
    socketreduceoutdispl[0] = 0;
    for(int p = 1; p < numproc+1; p++)
      socketreduceoutdispl[p] = socketreduceoutdispl[p-1] + socketreduceout[p-1];
    int **socketreducedispltemp = new int*[numproc];
    int **socketreduceindextemp = new int*[numproc];
    #pragma omp parallel for
    for(int socket = 0; socket < numsocket; socket++){
      for(int p = socket*numproc_socket; p < (socket+1)*numproc_socket; p++){
        int tempdispl = socketreduceoutdispl[p]-socketreduceoutdispl[socket*numproc_socket];
        socketreducedispltemp[p] = new int[socketreduceout[p]+1];
        socketreducedispltemp[p][0] = 0;
        for(int m = 1; m < socketreduceout[p]+1; m++)
          socketreducedispltemp[p][m] = socketreducedispltemp[p][m-1]+socketnz[socket][tempdispl+m-1];
        socketreduceindextemp[p] = new int[socketreducedispltemp[p][socketreduceout[p]]];
        for(int m = 0; m < socketreduceout[p]; m++)
          for(int n = 0; n < socketnz[socket][tempdispl+m]; n++)
            socketreduceindextemp[p][socketreducedispltemp[p][m]+n] = socketindex[socket][socketdispl[socket][tempdispl+m]+n]-rayrecvstart[socket*numproc_socket];
      }
      delete[] socketdispl[socket];
      delete[] socketindex[socket];
    }
    MPI_Request sendrequest[numproc];
    MPI_Request recvrequest[numproc];
    socketreduceinc = socketreduceout;
    socketreduceincdispl = socketreduceoutdispl;
    socketreduceout = new int[numproc];
    socketreduceoutdispl = new int[numproc+1];
    MPI_Alltoall(socketreduceinc,1,MPI_INT,socketreduceout,1,MPI_INT,MPI_COMM_WORLD);
    socketreduceoutdispl[0] = 0;
    for(int n = 1; n < numproc+1; n++)
      socketreduceoutdispl[n] = socketreduceoutdispl[n-1] + socketreduceout[n-1];
    int **displtemp = socketreducedispltemp;
    int **indextemp = socketreduceindextemp;
    for(int p = 0; p < numproc; p++){
      MPI_Isend(displtemp[p],socketreduceinc[p]+1,MPI_INT,p,0,MPI_COMM_WORLD,sendrequest+p);
      MPI_Isend(indextemp[p],displtemp[p][socketreduceinc[p]],MPI_INT,p,1,MPI_COMM_WORLD,recvrequest+p);
    }
    socketreducedispltemp = new int*[numproc];
    socketreduceindextemp = new int*[numproc];
    for(int p = 0; p < numproc; p++){
      socketreducedispltemp[p] = new int[socketreduceout[p]+1];
      MPI_Recv(socketreducedispltemp[p],socketreduceout[p]+1,MPI_INT,p,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      socketreduceindextemp[p] = new int[socketreducedispltemp[p][socketreduceout[p]]];
      MPI_Recv(socketreduceindextemp[p],socketreducedispltemp[p][socketreduceout[p]],MPI_INT,p,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    MPI_Waitall(numproc,sendrequest,MPI_STATUSES_IGNORE);
    MPI_Waitall(numproc,recvrequest,MPI_STATUSES_IGNORE);
    for(int p = 0; p < numproc; p++){
      delete[] displtemp[p];
      delete[] indextemp[p];
    }
    delete[] displtemp;
    delete[] indextemp;
    //RE-INDEX SOCKET SENDING SIDE
    int raynumouts[numproc_socket];
    MPI_Allgather(&raynumout,1,MPI_INT,raynumouts,1,MPI_INT,MPI_COMM_SOCKET);
    int raynumoutdispl[numproc_socket+1];
    raynumoutdispl[0] = 0;
    for(int p = 1; p < numproc_socket+1; p++)
      raynumoutdispl[p] = raynumoutdispl[p-1] + raynumouts[p-1];
    int *socketindexmaptemp = new int[raynumouts[myid_socket]];
    #pragma omp parallel for
    for(int n = 0; n < raynumouts[myid_socket]; n++)
      socketindexmaptemp[n] = raynumoutdispl[myid_socket]+n;
    int *socketindexmap= new int[raynumoutdispl[numproc_socket]];
    int reducebuffdispl[numproc+1];
    int recvtemp = 0;
    reducebuffdispl[0] = 0;
    for(int p = 0; p < numproc; p++){
      for(int precv = 0; precv < numproc_socket; precv++){
        MPI_Isend(&raysendcount[p],1,MPI_INT,precv,0,MPI_COMM_SOCKET,sendrequest+precv);
        MPI_Isend(socketindexmaptemp+raysendstart[p],raysendcount[p],MPI_INT,precv,1,MPI_COMM_SOCKET,recvrequest+precv);
      }
      for(int psend = 0; psend < numproc_socket; psend++){
        int recvcount = -1;
        MPI_Recv(&recvcount,1,MPI_INT,psend,0,MPI_COMM_SOCKET,MPI_STATUS_IGNORE);
        MPI_Recv(socketindexmap+recvtemp,recvcount,MPI_INT,psend,1,MPI_COMM_SOCKET,MPI_STATUS_IGNORE);
        recvtemp += recvcount;
      }
      reducebuffdispl[p+1] = recvtemp;
    }
    MPI_Waitall(numproc_socket,sendrequest,MPI_STATUSES_IGNORE);
    MPI_Waitall(numproc_socket,recvrequest,MPI_STATUSES_IGNORE);
    delete[] socketindexmaptemp;
    socketreducedispl = new int[socketreduceoutdispl[numproc]+1];
    socketreducedispl[0] = 0;
    for(int p = 0; p < numproc; p++)
      for(int m = 0; m < socketreduceout[p]; m++){
        int ind = socketreduceoutdispl[p]+m;
        int nz = socketreducedispltemp[p][m+1]-socketreducedispltemp[p][m];
        socketreducedispl[ind+1] = socketreducedispl[ind]+nz;
      }
    socketreduceindex = new int[socketreducedispl[socketreduceoutdispl[numproc]]];
    #pragma omp parallel for
    for(int p = 0; p < numproc; p++)
      for(int m = 0; m < socketreduceout[p]; m++){
        for(int n = socketreducedispltemp[p][m]; n < socketreducedispltemp[p][m+1]; n++){
          int oldindex = socketindexmap[reducebuffdispl[p]+socketreduceindextemp[p][n]];
          int newindex = socketreducedispl[socketreduceoutdispl[p]+m]+n-socketreducedispltemp[p][m];
          socketreduceindex[newindex] = oldindex;
        }
      }
    delete[] socketindexmap;
    int *socketrecvbufftag = new int[raynumoutdispl[numproc_socket]];
    #pragma omp parallel for
    for(int p = 0; p < numproc_socket; p++)
      for(int n = raynumoutdispl[p]; n < raynumoutdispl[p+1]; n++)
        socketrecvbufftag[n] = p;
    socketrecvcomm = new int[numproc_socket];
    #pragma omp parallel for
    for(int p = 0; p < numproc_socket; p++)
      socketrecvcomm[p] = 0;
    for(int m = 0; m < socketreduceoutdispl[numproc]; m++)
      for(int n = socketreducedispl[m]; n < socketreducedispl[m+1]; n++)
        socketrecvcomm[socketrecvbufftag[socketreduceindex[n]]]++;
    socketrecvcommdispl = new int[numproc_socket+1];
    socketrecvcommdispl[0] = 0;
    for(int p = 1; p < numproc_socket+1; p++)
      socketrecvcommdispl[p] = socketrecvcommdispl[p-1] + socketrecvcomm[p-1];
    #pragma omp parallel for
    for(int n = 0; n < raynumoutdispl[numproc_socket]; n++)
      socketrecvbufftag[n] = -1;
    #pragma omp parallel for
    for(int n = 0; n < socketreducedispl[socketreduceoutdispl[numproc]]; n++)
      socketrecvbufftag[socketreduceindex[n]] = 0;
    int *socketrecvcommap[numproc_socket];
    #pragma omp parallel for
    for(int p = 0; p < numproc_socket; p++){
      socketrecvcommap[p] = new int[socketrecvcomm[p]];
      int count = 0;
      for(int n = raynumoutdispl[p]; n < raynumoutdispl[p+1]; n++)
        if(socketrecvbufftag[n] > -1){
          socketrecvbufftag[n] = socketrecvcommdispl[p]+count;
          socketrecvcommap[p][count] = n-raynumoutdispl[p];
          count++;
        }
    }
    socketsendcomm = new int[numproc_socket];
    for(int p = 0; p < numproc_socket; p++){
      MPI_Isend(&socketrecvcomm[p],1,MPI_INT,p,0,MPI_COMM_SOCKET,sendrequest+p);
      MPI_Recv(&socketsendcomm[p],1,MPI_INT,p,0,MPI_COMM_SOCKET,MPI_STATUS_IGNORE);
    }
    MPI_Waitall(numproc_socket,sendrequest,MPI_STATUSES_IGNORE);
    socketsendcommdispl = new int[numproc_socket+1];
    socketsendcommdispl[0] = 0;
    for(int p = 1; p < numproc_socket+1; p++)
      socketsendcommdispl[p] = socketsendcommdispl[p-1] + socketsendcomm[p-1];
    socketsendmap = new int[socketsendcommdispl[numproc_socket]];
    for(int p = 0; p < numproc_socket; p++){
      MPI_Isend(socketrecvcommap[p],socketrecvcomm[p],MPI_INT,p,0,MPI_COMM_SOCKET,sendrequest+p);
      MPI_Recv(socketsendmap+socketsendcommdispl[p],socketsendcomm[p],MPI_INT,p,0,MPI_COMM_SOCKET,MPI_STATUS_IGNORE);
    }
    MPI_Waitall(numproc_socket,sendrequest,MPI_STATUSES_IGNORE);
    #pragma omp parallel for
    for(int p = 0; p < numproc_socket; p++)
      delete[] socketrecvcommap[p];
    #pragma omp parallel for
    for(int m = 0; m < socketreduceoutdispl[numproc]; m++)
      for(int n = socketreducedispl[m]; n < socketreducedispl[m+1]; n++)
        socketreduceindex[n] = socketrecvbufftag[socketreduceindex[n]];
    delete[] socketrecvbufftag;
  }
  //NODE REDUCTION MAPS
  {
    int *nodemap = new int[socketreduceincdispl[numproc]];
    int noderayinc[numnode];
    #pragma omp parallel for
    for(int node = 0; node < numnode; node++){
      for(int p = node*numproc_node; p < (node+1)*numproc_node; p++)
        for(int n = socketreduceincdispl[p]; n < socketreduceincdispl[p+1]; n++)
          nodemap[n] = node;
      noderayinc[node] = 0;
    }
    for(int m = 0; m < mynumray; m++){
      int count[numnode];
      for(int node = 0; node < numnode; node++)
        count[node] = 0;
      for(int n = socketraydispl[m]; n < socketraydispl[m+1]; n++)
        count[nodemap[socketrayindex[n]]]++;
      for(int node = 0; node < numnode; node++)
        if(count[node])
          noderayinc[node]++;
    }
    int noderayincdispl[numnode+1];
    int *nodenz[numnode];
    noderayincdispl[0] = 0;
    for(int node = 0; node < numnode; node++){
      noderayincdispl[node+1] = noderayincdispl[node] + noderayinc[node];
      nodenz[node] = new int[noderayinc[node]];
      noderayinc[node] = 0;
    }
    noderayoutall = noderayincdispl[numnode];
    MPI_Allreduce(MPI_IN_PLACE,&noderayoutall,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    if(myid==0)printf("  INTER-NODE COMM: %ld (%f MB) %f%% saving (%f%% additional saving)\n",noderayoutall,noderayoutall*sizeof(VECPREC)/1.0e6,(raynumoutall-noderayoutall)/(double)raynumoutall*100,(socketrayoutall-noderayoutall)/(double)socketrayoutall*100);
    for(int m = 0; m < mynumray; m++){
      int count[numnode];
      for(int node = 0; node < numnode; node++)
        count[node] = 0;
      for(int n = socketraydispl[m]; n < socketraydispl[m+1]; n++)
        count[nodemap[socketrayindex[n]]]++;
      for(int node = 0; node < numnode; node++)
        if(count[node]){
          nodenz[node][noderayinc[node]] = count[node];
          noderayinc[node]++;
        }
    }
    int *nodedispl[numnode];
    int *nodeindex[numnode];
    #pragma omp parallel for
    for(int node = 0; node < numnode; node++){
      nodedispl[node] = new int[noderayinc[node]+1];
      nodedispl[node][0] = 0;
      for(int k = 1; k < noderayinc[node]+1; k++)
        nodedispl[node][k] = nodedispl[node][k-1] + nodenz[node][k-1];
      nodeindex[node] = new int[nodedispl[node][noderayinc[node]]];
      noderayinc[node] = 0;
    }
    for(int m = 0; m < mynumray; m++){
      int count[numnode];
      for(int node = 0; node < numnode; node++)
        count[node] = 0;
      for(int n = socketraydispl[m]; n < socketraydispl[m+1]; n++){
        int index = socketrayindex[n];
        int node = nodemap[index];
        nodeindex[node][nodedispl[node][noderayinc[node]]+count[node]] = index;
        count[node]++;
      }
      for(int node = 0; node < numnode; node++)
        if(count[node])
          noderayinc[node]++;
    }
    delete[] nodemap;
    //FIND NODE REDUCTION MAPPING RECEIVING SIDE
    int *noderaymap = new int[socketreduceincdispl[numproc]];
    #pragma omp parallel for
    for(int m = 0; m < mynumray; m++)
      for(int n = socketraydispl[m]; n < socketraydispl[m+1]; n++)
        noderaymap[socketrayindex[n]] = m;
    int *nodeinvmap = new int[noderayincdispl[numnode]];
    #pragma omp parallel for
    for(int node = 0; node < numnode; node++)
      for(int m = 0; m < noderayinc[node]; m++)
        for(int n = nodedispl[node][m]; n < nodedispl[node][m+1]; n++)
          nodeinvmap[noderayincdispl[node]+m] = noderaymap[nodeindex[node][n]];
    delete[] noderaymap;
    int *noderaynz = new int[mynumray];
    #pragma omp parallel for
    for(int n = 0; n < mynumray; n++)
      noderaynz[n] = 0;
    for(int n = 0; n < noderayincdispl[numnode]; n++)
      noderaynz[nodeinvmap[n]]++;
    noderaydispl = new int[mynumray+1];
    noderaydispl[0] = 0;
    for(int m = 1; m < mynumray+1; m++){
      noderaydispl[m] = noderaydispl[m-1] + noderaynz[m-1];
      noderaynz[m-1] = 0;
    }
    noderayindex = new int[noderaydispl[mynumray]];
    for(int n = 0; n < noderayincdispl[numnode]; n++){
      int ray = nodeinvmap[n];
      noderayindex[noderaydispl[ray]+noderaynz[ray]] = n;
      noderaynz[ray]++;
    }
    delete[] nodeinvmap;
    delete[] noderaynz;
    //FIND NODE REDUCTION MAPPING SENDING SIDE
    nodereduceout = new int[numproc];
    #pragma omp parallel for
    for(int p = 0; p < numproc; p++)
      nodereduceout[p] = 0;
    #pragma omp parallel for
    for(int node = 0; node < numnode; node++)
      nodereduceout[node*numproc_node+myid_node] = noderayinc[node];
    /*#pragma omp parallel for
    for(int node = 0; node < numnode; node++){
      int nodeouttemp = (noderayinc[node]/numproc_node)*numproc_node;
      for(int p = node*numproc_node; p < (node+1)*numproc_node; p++){
        nodereduceout[p] = noderayinc[node]/numproc_node;
        if(nodeouttemp < noderayinc[node]){
          nodereduceout[p]++;
          nodeouttemp++;
        }
      }
    }*/
    nodereduceoutdispl = new int[numproc+1];
    nodereduceoutdispl[0] = 0;
    for(int p = 1; p < numproc+1; p++)
      nodereduceoutdispl[p] = nodereduceoutdispl[p-1] + nodereduceout[p-1];
    int **nodereducedispltemp = new int*[numproc];
    int **nodereduceindextemp = new int*[numproc];
    #pragma omp parallel for
    for(int node = 0; node < numnode; node++){
      for(int p = node*numproc_node; p < (node+1)*numproc_node; p++){
        int tempdispl = nodereduceoutdispl[p]-nodereduceoutdispl[node*numproc_node];
        nodereducedispltemp[p] = new int[nodereduceout[p]+1];
        nodereducedispltemp[p][0] = 0;
        for(int m = 1; m < nodereduceout[p]+1; m++)
          nodereducedispltemp[p][m] = nodereducedispltemp[p][m-1]+nodenz[node][tempdispl+m-1];
        nodereduceindextemp[p] = new int[nodereducedispltemp[p][nodereduceout[p]]];
        for(int m = 0; m < nodereduceout[p]; m++)
          for(int n = 0; n < nodenz[node][tempdispl+m]; n++)
            nodereduceindextemp[p][nodereducedispltemp[p][m]+n] = nodeindex[node][nodedispl[node][tempdispl+m]+n]-socketreduceincdispl[node*numproc_node];
      }
      delete[] nodedispl[node];
      delete[] nodeindex[node];
    }
    MPI_Request sendrequest[numproc];
    MPI_Request recvrequest[numproc];
    nodereduceinc = nodereduceout;
    nodereduceincdispl = nodereduceoutdispl;
    nodereduceout = new int[numproc];
    nodereduceoutdispl = new int[numproc+1];
    MPI_Alltoall(nodereduceinc,1,MPI_INT,nodereduceout,1,MPI_INT,MPI_COMM_WORLD);
    nodereduceoutdispl[0] = 0;
    for(int n = 1; n < numproc+1; n++)
      nodereduceoutdispl[n] = nodereduceoutdispl[n-1] + nodereduceout[n-1];
    int **displtemp = nodereducedispltemp;
    int **indextemp = nodereduceindextemp;
    for(int p = 0; p < numproc; p++){
      MPI_Isend(displtemp[p],nodereduceinc[p]+1,MPI_INT,p,0,MPI_COMM_WORLD,sendrequest+p);
      MPI_Isend(indextemp[p],displtemp[p][nodereduceinc[p]],MPI_INT,p,1,MPI_COMM_WORLD,recvrequest+p);
    }
    nodereducedispltemp = new int*[numproc];
    nodereduceindextemp = new int*[numproc];
    for(int p = 0; p < numproc; p++){
      nodereducedispltemp[p] = new int[nodereduceout[p]+1];
      MPI_Recv(nodereducedispltemp[p],nodereduceout[p]+1,MPI_INT,p,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      nodereduceindextemp[p] = new int[nodereducedispltemp[p][nodereduceout[p]]];
      MPI_Recv(nodereduceindextemp[p],nodereducedispltemp[p][nodereduceout[p]],MPI_INT,p,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
    MPI_Waitall(numproc,sendrequest,MPI_STATUSES_IGNORE);
    MPI_Waitall(numproc,recvrequest,MPI_STATUSES_IGNORE);
    for(int p = 0; p < numproc; p++){
      delete[] displtemp[p];
      delete[] indextemp[p];
    }
    delete[] displtemp;
    delete[] indextemp;
    //RE-INDEX NODE SENDING SIDE
    int raynumouts[numproc_node];
    MPI_Allgather(&socketreduceoutdispl[numproc],1,MPI_INT,raynumouts,1,MPI_INT,MPI_COMM_NODE);
    int raynumoutdispl[numproc_node+1];
    raynumoutdispl[0] = 0;
    for(int p = 1; p < numproc_node+1; p++)
      raynumoutdispl[p] = raynumoutdispl[p-1] + raynumouts[p-1];
    int *nodeindexmaptemp = new int[raynumouts[myid_node]];
    #pragma omp parallel for
    for(int n = 0; n < raynumouts[myid_node]; n++)
      nodeindexmaptemp[n] = raynumoutdispl[myid_node]+n;
    int *nodeindexmap= new int[raynumoutdispl[numproc_node]];
    int reducebuffdispl[numproc+1];
    int recvtemp = 0;
    reducebuffdispl[0] = 0;
    for(int p = 0; p < numproc; p++){
      for(int precv = 0; precv < numproc_node; precv++){
        MPI_Isend(&socketreduceout[p],1,MPI_INT,precv,0,MPI_COMM_NODE,sendrequest+precv);
        MPI_Isend(nodeindexmaptemp+socketreduceoutdispl[p],socketreduceout[p],MPI_INT,precv,1,MPI_COMM_NODE,recvrequest+precv);
      }
      for(int psend = 0; psend < numproc_node; psend++){
        int recvcount = -1;
        MPI_Recv(&recvcount,1,MPI_INT,psend,0,MPI_COMM_NODE,MPI_STATUS_IGNORE);
        MPI_Recv(nodeindexmap+recvtemp,recvcount,MPI_INT,psend,1,MPI_COMM_NODE,MPI_STATUS_IGNORE);
        recvtemp += recvcount;
      }
      reducebuffdispl[p+1] = recvtemp; 
    }
    MPI_Waitall(numproc_node,sendrequest,MPI_STATUSES_IGNORE);
    MPI_Waitall(numproc_node,recvrequest,MPI_STATUSES_IGNORE);
    delete[] nodeindexmaptemp;
    nodereducedispl = new int[nodereduceoutdispl[numproc]+1];
    nodereducedispl[0] = 0;
    for(int p = 0; p < numproc; p++)
      for(int m = 0; m < nodereduceout[p]; m++){
        int ind = nodereduceoutdispl[p]+m;
        int nz = nodereducedispltemp[p][m+1]-nodereducedispltemp[p][m];
        nodereducedispl[ind+1] = nodereducedispl[ind]+nz;
      }
    nodereduceindex = new int[nodereducedispl[nodereduceoutdispl[numproc]]];
    #pragma omp parallel for
    for(int p = 0; p < numproc; p++)
      for(int m = 0; m < nodereduceout[p]; m++){
        for(int n = nodereducedispltemp[p][m]; n < nodereducedispltemp[p][m+1]; n++){
          int oldindex = nodeindexmap[reducebuffdispl[p]+nodereduceindextemp[p][n]];
          int newindex = nodereducedispl[nodereduceoutdispl[p]+m]+n-nodereducedispltemp[p][m];
          nodereduceindex[newindex] = oldindex;
        }
      }
    delete[] nodeindexmap;
    int *noderecvbufftag = new int[raynumoutdispl[numproc_node]];
    #pragma omp parallel for
    for(int p = 0; p < numproc_node; p++)
      for(int n = raynumoutdispl[p]; n < raynumoutdispl[p+1]; n++)
        noderecvbufftag[n] = p;
    noderecvcomm = new int[numproc_node];
    #pragma omp parallel for
    for(int p = 0; p < numproc_node; p++)
      noderecvcomm[p] = 0;
    for(int m = 0; m < nodereduceoutdispl[numproc]; m++)
      for(int n = nodereducedispl[m]; n < nodereducedispl[m+1]; n++)
        noderecvcomm[noderecvbufftag[nodereduceindex[n]]]++;
    noderecvcommdispl = new int[numproc_node+1];
    noderecvcommdispl[0] = 0;
    for(int p = 1; p < numproc_node+1; p++)
      noderecvcommdispl[p] = noderecvcommdispl[p-1] + noderecvcomm[p-1];
    #pragma omp parallel for
    for(int n = 0; n < raynumoutdispl[numproc_node]; n++)
      noderecvbufftag[n] = -1;
    #pragma omp parallel for
    for(int n = 0; n < nodereducedispl[nodereduceoutdispl[numproc]]; n++)
      noderecvbufftag[nodereduceindex[n]] = 0;
    int *noderecvcommap[numproc_node];
    #pragma omp parallel for
    for(int p = 0; p < numproc_node; p++){
      noderecvcommap[p] = new int[noderecvcomm[p]];
      int count = 0;
      for(int n = raynumoutdispl[p]; n < raynumoutdispl[p+1]; n++)
        if(noderecvbufftag[n] > -1){
          noderecvbufftag[n] = noderecvcommdispl[p]+count;
          noderecvcommap[p][count] = n-raynumoutdispl[p];
          count++;
        }
    }
    nodesendcomm = new int[numproc_node];
    for(int p = 0; p < numproc_node; p++){
      MPI_Isend(&noderecvcomm[p],1,MPI_INT,p,0,MPI_COMM_NODE,sendrequest+p);
      MPI_Recv(&nodesendcomm[p],1,MPI_INT,p,0,MPI_COMM_NODE,MPI_STATUS_IGNORE);
    }
    MPI_Waitall(numproc_node,sendrequest,MPI_STATUSES_IGNORE);
    nodesendcommdispl = new int[numproc_node+1];
    nodesendcommdispl[0] = 0;
    for(int p = 1; p < numproc_node+1; p++)
      nodesendcommdispl[p] = nodesendcommdispl[p-1] + nodesendcomm[p-1];
    nodesendmap = new int[nodesendcommdispl[numproc_node]];
    for(int p = 0; p < numproc_node; p++){
      MPI_Isend(noderecvcommap[p],noderecvcomm[p],MPI_INT,p,0,MPI_COMM_NODE,sendrequest+p);
      MPI_Recv(nodesendmap+nodesendcommdispl[p],nodesendcomm[p],MPI_INT,p,0,MPI_COMM_NODE,MPI_STATUS_IGNORE);
    }
    MPI_Waitall(numproc_node,sendrequest,MPI_STATUSES_IGNORE);
    #pragma omp parallel for
    for(int p = 0; p < numproc_node; p++)
      delete[] noderecvcommap[p];
    #pragma omp parallel for
    for(int m = 0; m < nodereduceoutdispl[numproc]; m++)
      for(int n = nodereducedispl[m]; n < nodereducedispl[m+1]; n++)
        nodereduceindex[n] = noderecvbufftag[nodereduceindex[n]];
    delete[] noderecvbufftag;
  }
  {
    socketpackmap = new int[socketsendcommdispl[numproc_socket]*FFACTOR];
    #pragma omp parallel for
    for(int p = 0; p < numproc_socket; p++)
      for(int m = 0; m < socketsendcomm[p]; m++)
        for(int s = 0; s < FFACTOR; s++){
          int ind = s*socketsendcommdispl[numproc_socket]+socketsendmap[socketsendcommdispl[p]+m];
          int index = socketsendcommdispl[p]*FFACTOR+s*socketsendcomm[p]+m;
          socketpackmap[ind] = index;
        }
    socketunpackmap = new int[socketrecvcommdispl[numproc_socket]*FFACTOR];
    #pragma omp parallel for
    for(int p = 0; p < numproc_socket; p++)
      for(int m = 0; m < socketrecvcomm[p]; m++)
        for(int s = 0; s < FFACTOR; s++){
          int index = socketrecvcommdispl[p]*FFACTOR+s*socketrecvcomm[p]+m;
          int ind = s*socketrecvcommdispl[numproc_socket]+socketrecvcommdispl[p]+m;
          socketunpackmap[ind] = index;
        }
    nodepackmap = new int[nodesendcommdispl[numproc_node]*FFACTOR];
    #pragma omp parallel for
    for(int p = 0; p < numproc_node; p++)
      for(int m = 0; m < nodesendcomm[p]; m++)
        for(int s = 0; s < FFACTOR; s++){
          int ind = s*nodesendcommdispl[numproc_node]+nodesendmap[nodesendcommdispl[p]+m];
          int index = nodesendcommdispl[p]*FFACTOR+s*nodesendcomm[p]+m;
          nodepackmap[ind] = index;
        }
    nodeunpackmap = new int[noderecvcommdispl[numproc_node]*FFACTOR];
    #pragma omp parallel for
    for(int p = 0; p < numproc_node; p++)
      for(int m = 0; m < noderecvcomm[p]; m++)
        for(int s = 0; s < FFACTOR; s++){
          int index = noderecvcommdispl[p]*FFACTOR+s*noderecvcomm[p]+m;
          int ind = s*noderecvcommdispl[numproc_node]+noderecvcommdispl[p]+m;
          nodeunpackmap[ind] = index;
        }
    raypackmap = new int[nodereduceoutdispl[numproc]*FFACTOR];
    #pragma omp parallel for
    for(int p = 0; p < numproc; p++)
      for(int m = 0; m < nodereduceout[p]; m++)
        for(int s = 0; s < FFACTOR; s++){
          int ind = s*nodereduceoutdispl[numproc]+nodereduceoutdispl[p]+m;
          int index = nodereduceoutdispl[p]*FFACTOR+s*nodereduceout[p]+m;
          raypackmap[ind] = index;
        }
    rayunpackmap = new int[nodereduceincdispl[numproc]*FFACTOR];
    #pragma omp parallel for
    for(int p = 0; p < numproc; p++)
      for(int m = 0; m < nodereduceinc[p]; m++)
        for(int s = 0; s < FFACTOR; s++){
          int index = nodereduceincdispl[p]*FFACTOR+s*nodereduceinc[p]+m;
          int ind = s*nodereduceincdispl[numproc]+nodereduceincdispl[p]+m;
          rayunpackmap[ind] = index;
        }
  }
}

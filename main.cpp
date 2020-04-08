#include "vars.h"

//TIMERS & COUNTERS
double ptime = 0;
double pktime = 0;
double pcstime = 0;
double pcntime = 0;
double pcrtime = 0;
double pchtime = 0;
double pmtime = 0;
double prtime = 0;
double btime = 0;
double bktime = 0;
double bcstime = 0;
double bcntime = 0;
double bcrtime = 0;
double bchtime = 0;
double bmtime = 0;
double brtime = 0;
double rtime = 0;
double iotime = 0;
int numproj = 0;
int numback = 0;

int numx; //X SIZE OF DOMAIN
int numy; //Y SIZE OF DOMAIN
int numt; //NUMBER OF THETA
int numr; //NUMBER OF RHO
int numslice; //NUMBER OF SLICES
int startslice; //START SLICE INDEX 0 BASE
int batchsize; //SLICE PER BATCH
int numbatch; //NUMBER OF BATCHES

double xstart; //X START OF DOMAIN
double ystart; //Y START OF DOMAIN
double pixsize; //PIXEL SIZE
double rhostart; //RHO START
double raylength; //RAYLENGTH

char *sinfile; //SINOGRAM FILE
char *thefile; //THETA FILE
char *outfile; //OUTPUT FILE
int numiter; //NUMBER OF ITERATIONS

int spatsize; //SPATIAL TILE SIZE
int specsize; //SPECTRAL TILE SIZE
int numxtile; //NUMBER OF X TILES
int numytile; //NUMBER OF Y TILES
int numttile; //NUMBER OF THETA TILES
int numrtile; //NUMBER OF RHO TILES
int numpix; //NUMBER OF PIXELS (EXTENDED)
int numray; //NUMBER OF RAYS (EXTENDED)
int numspattile; //NUMBER OF SPATIAL TILES
int numspectile; //NUMBER OF SPECTRAL TILES

int proj_blocksize;
int proj_buffsize;
int back_blocksize;
int back_buffsize;

int raynuminc;
int raynumout;
int mynumray;
int mynumpix;

int numthreads;
int numproc;
int myid;
MPI_Comm MPI_COMM_SOCKET;
int numproc_socket;
int myid_socket;
int numsocket;
MPI_Comm MPI_COMM_NODE;
int numproc_node;
int myid_node;
int numnode;

int main(int argc, char** argv){

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  #pragma omp parallel
  if(omp_get_thread_num()==0)numthreads=omp_get_num_threads();

  MPI_Barrier(MPI_COMM_WORLD);
  double timetot = MPI_Wtime();

  //SCANNING GEOMETRY DATA
  char *chartemp;
  chartemp = getenv("NUMTHE");
  numt = atoi(chartemp);
  chartemp = getenv("NUMRHO");
  numr = atoi(chartemp);
  //chartemp = getenv("NUMX");
  numx = numr;//atoi(chartemp);
  //chartemp = getenv("NUMY");
  numy = numx;//atoi(chartemp);
  chartemp = getenv("NUMSLICE");
  numslice = atoi(chartemp);
  chartemp = getenv("STARTSLICE");
  startslice = atoi(chartemp);
  chartemp = getenv("BATCHSIZE");
  batchsize = atoi(chartemp);

  chartemp = getenv("PIXSIZE");
  pixsize = atof(chartemp);
  chartemp = getenv("XSTART");
  xstart = atof(chartemp);
  //chartemp = getenv("YSTART");
  ystart = xstart;//atof(chartemp);
  chartemp = getenv("RHOSTART");
  rhostart = atof(chartemp);

  chartemp = getenv("NUMITER");
  numiter = atoi(chartemp);

  chartemp = getenv("SPATSIZE");
  spatsize = atof(chartemp);
  chartemp = getenv("SPECSIZE");
  specsize = atof(chartemp);

  chartemp = getenv("PROJBLOCK");
  proj_blocksize = atoi(chartemp);
  chartemp = getenv("BACKBLOCK");
  back_blocksize = atoi(chartemp);
  chartemp = getenv("PROJBUFF");
  proj_buffsize = atoi(chartemp);
  chartemp = getenv("BACKBUFF");
  back_buffsize = atoi(chartemp);

  sinfile = getenv("SINFILE");
  thefile = getenv("THEFILE");
  outfile = getenv("OUTFILE");
  chartemp = getenv("PROCPERNODE");
  numproc_node = atoi(chartemp);
  chartemp = getenv("PROCPERSOCKET");
  numproc_socket= atoi(chartemp);

  //FIND NUMBER OF TILES
  numxtile = numx/spatsize;
  if(numx%spatsize)numxtile++;
  numytile = numy/spatsize;
  if(numy%spatsize)numytile++;
  numrtile = numr/specsize;
  if(numr%specsize)numrtile++;
  numttile = numt/specsize;
  if(numt%specsize)numttile++;
  numspattile = numxtile*numytile;
  numspectile = numrtile*numttile;
  numpix = numspattile*pow(spatsize,2);
  numray = numspectile*pow(specsize,2);
  if(numx > numy)raylength = 2*numx;
  else raylength = 2*numy;
  proj_buffsize = proj_buffsize*1024/sizeof(VECPREC);
  back_buffsize = back_buffsize*1024/sizeof(VECPREC);
  numbatch = numslice/batchsize;
  //SOCKETS AND NODES
  MPI_Comm_split(MPI_COMM_WORLD,myid/numproc_socket,myid,&MPI_COMM_SOCKET);
  MPI_Comm_rank(MPI_COMM_SOCKET,&myid_socket);
  MPI_Comm_size(MPI_COMM_SOCKET,&numproc_socket);
  numsocket = numproc/numproc_socket;
  MPI_Comm_split(MPI_COMM_WORLD,myid/numproc_node,myid,&MPI_COMM_NODE);
  MPI_Comm_rank(MPI_COMM_NODE,&myid_node);
  MPI_Comm_size(MPI_COMM_NODE,&numproc_node);
  numnode = numproc/numproc_node;

  int myids_socket[numproc];
  int myids_node[numproc];
  MPI_Allgather(&myid_socket,1,MPI_INT,myids_socket,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&myid_node,1,MPI_INT,myids_node,1,MPI_INT,MPI_COMM_WORLD);
  if(myid==0)
    for(int p = 0; p < numproc; p++)
      printf("myid %d myid_socket %d myid_node %d\n",p,myids_socket[p],myids_node[p]);
  int myidss_socket[numproc_socket];
  int myidss_node[numproc_node];
  MPI_Allgather(&myid,1,MPI_INT,myidss_socket,1,MPI_INT,MPI_COMM_SOCKET);
  MPI_Allgather(&myid,1,MPI_INT,myidss_node,1,MPI_INT,MPI_COMM_NODE);
  if(myid==0){
    for(int p = 0; p < numproc_socket; p++)
      printf("myid_socket %d myid %d\n",p,myidss_socket[p]);
    for(int p = 0; p < numproc_node; p++)
      printf("myid_node %d myid %d\n",p,myidss_node[p]);
  }
  
  //PRINT DATA
  if(myid==0){
    printf("  NUMBER OF RAYS (THE x RHO): %d (%d x %d)\n",numt*numr,numt,numr);
    printf("    NUMBER OF PIXELS (X x Y): %d (%d x %d)\n",numx*numy,numx,numy);
    printf("         NUM. SLICES (B x S): %d (%d x %d)\n",numslice,numbatch,batchsize);
    printf("                 START SLICE: %d\n",startslice);
    printf("                  BATCH SIZE: %d FUSE FACTOR %d\n",batchsize,FFACTOR);
    printf("\n");
    printf("     SPATIAL / SPECTRAL  TILE SIZE: %d (%d x %d) / %d (%d x %d)\n",spatsize*spatsize,spatsize,spatsize,specsize*specsize,specsize,specsize);
    printf("NUMBER OF SPATIAL / SPECTRAL TILES: %d (%d x %d) / %d (%d x %d)\n",numspattile,numxtile,numytile,numspectile,numttile,numrtile);
    printf("  NUMBER OF EXTENDED PIXELS / RAYS: %d (%d x %d) %f / %d (%d x %d) %f\n",numpix,numxtile*spatsize,numytile*spatsize,numpix/(double)(numx*numy),numray,numttile*specsize,numrtile*specsize,numray/(double)(numt*numr));
    printf("\n");
    printf("    NUMBER OF PROCESSES : %d\n",numproc);
    printf("   NUMBER OF THRD./PROC.: %d\n",numthreads);
    printf("\n");
    printf("             BLOCK SIZE : %d / %d\n",proj_blocksize,back_blocksize);
    printf("NUM. BLOCKS (PER PROC.) : %d (%d) / %d (%d)\n",numray/proj_blocksize,numray/proj_blocksize/numproc,numpix/back_blocksize,numpix/back_blocksize/numproc);
    printf("            BUFFER SIZE : %d (%f KB) / %d (%f KB)\n",proj_buffsize,proj_buffsize*sizeof(VECPREC)/1024.0,back_buffsize,back_buffsize*sizeof(VECPREC)/1024.0);
  }
  proj_buffsize = proj_buffsize/FFACTOR;
  back_buffsize = back_buffsize/FFACTOR;
  if(myid==0){
    printf("  EFFECTIVE BUFFER SIZE : %d (%f KB) / %d (%f KB)\n",proj_buffsize*FFACTOR,proj_buffsize*(int)sizeof(VECPREC)/1024.0*FFACTOR,back_buffsize*FFACTOR,back_buffsize*sizeof(VECPREC)/1024.0*FFACTOR);
    printf("   BUFFER SIZE PER SLICE: %d (%f KB) / %d (%f KB)\n",proj_buffsize,proj_buffsize*(int)sizeof(VECPREC)/1024.0,back_buffsize,back_buffsize*sizeof(VECPREC)/1024.0);
    printf("\n");
    printf("INTEGER: %ld, DOUBLE: %ld, LONG: %ld, SHORT: %ld, MATRIX %ld, POINTER: %ld\n",sizeof(int),sizeof(double),sizeof(long),sizeof(unsigned short),sizeof(matrix),sizeof(float*));
    printf("\n");
    printf("X & Y START   : (%f %f)\n",xstart,ystart);
    printf("RHO START     : %f\n",rhostart);
    printf("PIXEL SIZE    : %f\n",pixsize);
    printf("RAY LENGTH    : %f\n",raylength);
    printf("NUMBER OF ITERATIONS: %d\n",numiter);
    printf("SINOGRAM FILE : %s\n",sinfile);
    printf("   THETA FILE : %s\n",thefile);
    printf("  OUTPUT FILE : %s\n",outfile);
    printf("\n");
    printf("NUMBER OF PROCS PER SOCKET: %d PER NODE: %d\n",numproc_socket,numproc_node);
    printf("NUMBER OF NODES: %d SOCKETS: %d PROCS: %d\n",numnode,numsocket,numproc);
    printf("\n");
    #ifdef MIXED
    printf("MIXED PRECISION ON\n");
    #else
    printf("MIXED PRECISION OFF\n");
    #endif
    #ifdef MATRIX
    printf("MATRIX STRUCTURE ON\n");
    #else
    printf("MATRIX STRUCTURE OFF\n");
    #endif
    #ifdef OVERLAP
    printf("COMMUNICATIONS OVERLAPPED\n");
    #else
    printf("COMMUNICATIONS SYNCHRONIZED\n");
    #endif
    printf("\n");
  }

  double timep = MPI_Wtime();
  //PREPROCESSING
  preproc();
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("PREPROCESSING TIME: %e\n",MPI_Wtime()-timep);

  double *obj_d;//OBJECT
  double *gra_d;//GRADIENT
  double *dir_d;//DIRECTION
  double *res_d;//RESIDUE
  double *ray_d;//RAYSUM
  double *obj_h;//OBJECT
  double *res_h;//RESIDUE

  setup_gpu(&obj_d,&gra_d,&dir_d,&res_d,&ray_d,&obj_h,&res_h);

  double backscale = 1.0;
  double projscale = 1.0;
  extern double proj_rowmax;
  extern double back_rowmax;
  float *pixsendbuff = new float[mynumpix*batchsize];
  float *pixrecvbuff;
  float *objwritebuff;
  if(myid == 0){
    pixrecvbuff = new float[(long)numpix*batchsize];
    objwritebuff = new float[(long)numx*numy*batchsize];
    printf("\n");
    printf("PIXRECVBUFF: %ld (%f GB)\n",(long)numpix*batchsize,sizeof(float)*numpix*batchsize/1.0e9);
    printf("OBJWRITEBUFF: %ld (%f GB)\n",(long)numx*numy*batchsize,sizeof(float)*numx*numy*batchsize/1.0e9);
  }
  float *raysendbuff;
  float *rayrecvbuff = new float[mynumray*batchsize];
  float *mesreadbuff;
  if(myid == 0){
    raysendbuff = new float[(long)numray*batchsize];
    mesreadbuff = new float[(long)numt*numr*batchsize];
    printf("RAYSENDBUFF: %ld (%f GB)\n",(long)numray*batchsize,sizeof(float)*numray*batchsize/1.0e9);
    printf("MESREADBUFF: %ld (%f GB)\n",(long)numt*numr*batchsize,sizeof(float)*numt*numr*batchsize/1.0e9);
  }
  if(myid==0)printf("INPUT FILE: %s\n",sinfile);
  if(myid==0)printf("OUTPUT FILE: %s\n",outfile);
  FILE *inputf;
  FILE *outputf;
  if(myid==0)inputf = fopen(sinfile,"rb");
  if(myid==0)fseek(inputf,sizeof(float)*startslice*numr*numt,SEEK_SET);
  if(myid==0)outputf = fopen(outfile,"wb");
  if(myid==0)printf("\nCONJUGATE-GRADIENT OPTIMIZATION\n");
  MPI_Barrier(MPI_COMM_WORLD);
  double time = 0;
  rtime = MPI_Wtime();
  for(int batch = 0; batch < numbatch; batch++){
    if(myid==0)printf("BATCH %d\n",batch);
    //READ BATCH
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime();
    {
      //READ START
      extern int *numrays;
      extern int *raystart;
      extern long *mesglobalind;
      MPI_Request sendrequest[numproc];
      if(myid==0){
        double readtime = MPI_Wtime();
        fread(mesreadbuff,sizeof(float),(long)numr*numt*batchsize,inputf);
        readtime = MPI_Wtime()-readtime;
        printf("READ TIME %e s (%f GB/s)\n",readtime,sizeof(float)*numr*numt*batchsize/readtime/1.0e9);
        #pragma omp parallel for
        for(long n = 0; n < (long)numray*batchsize; n++){
          long ind = mesglobalind[n];
          if(ind > -1)raysendbuff[n] = mesreadbuff[ind];
          else raysendbuff[n] = 0;
        }
        for(int p = 0; p < numproc; p++)
          MPI_Issend(raysendbuff+(long)raystart[p]*batchsize,numrays[p]*batchsize,MPI_FLOAT,p,0,MPI_COMM_WORLD,sendrequest+p);
      }
      MPI_Recv(rayrecvbuff,mynumray*batchsize,MPI_FLOAT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      #pragma omp parallel for
      for(int n = 0; n < mynumray*batchsize; n++)
        res_h[n] = rayrecvbuff[n];
      //READ COMPLETE
      copyH2D_kernel(res_d,res_h,mynumray*batchsize);
      init_kernel(obj_d,mynumpix*batchsize);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    iotime += MPI_Wtime()-time;
    //FIND GRADIENT
    double resnorm = dot_kernel(res_d,res_d,mynumray*batchsize);
    double mesnorm = resnorm;
    double resmax = max_kernel(res_h,mynumray*batchsize);
    backscale = 64.0e3/(resmax*back_rowmax);
    backproject(gra_d,res_d,backscale);
    double gradnorm = dot_kernel(gra_d,gra_d,mynumpix*batchsize);
    double dirmax = max_kernel(gra_d,mynumpix*batchsize);
    double objnorm = 0.0;
    if(myid==0)printf("iter: %d resnorm: %e relnorm: %e resmax: %e dirmax: %e objnorm: %e bscale: %e\n",0,resnorm,1.0,resmax,dirmax,objnorm,backscale);
    //SAVE DIRECTION
    double oldgradnorm = gradnorm;
    copyD2D_kernel(dir_d,gra_d,mynumpix*batchsize);
    //START ITERATIONS
    for(int iter = 1; iter <= numiter; iter++){
      //PROJECT DIRECTION
      projscale = 64.0e3/(dirmax*proj_rowmax);
      project(ray_d,dir_d,projscale);
      //FIND STEP SIZE
      double temp1 = dot_kernel(res_d,ray_d,mynumray*batchsize);
      double temp2 = dot_kernel(ray_d,ray_d,mynumray*batchsize);
      //STEP SIZE
      double alpha = temp1/temp2;
      saxpy_kernel(obj_d,obj_d,alpha,dir_d,mynumpix*batchsize);
      objnorm = dot_kernel(obj_d,obj_d,mynumpix*batchsize);
      //FIND RESIDUAL ERROR
      saxpy_kernel(res_d,res_d,-alpha,ray_d,mynumray*batchsize);
      resnorm = dot_kernel(res_d,res_d,mynumray*batchsize);
      resmax = max_kernel(res_d,mynumray*batchsize);
      //FIND GRADIENT
      backscale = 64.0e3/(resmax*back_rowmax);
      backproject(gra_d,res_d,backscale);
      gradnorm = dot_kernel(gra_d,gra_d,mynumpix*batchsize);
      if(myid==0)printf("iter: %d resnorm: %e relnorm: %e resmax: %e dirmax: %e objnorm: %e bscale: %e pscale %e\n",iter,resnorm,resnorm/mesnorm,resmax,dirmax,objnorm,backscale,projscale);
      //UPDATE DIRECTION
      double beta = gradnorm/oldgradnorm;
      oldgradnorm = gradnorm;
      saxpy_kernel(dir_d,gra_d,beta,dir_d,mynumpix*batchsize);
      dirmax = max_kernel(dir_d,mynumpix*batchsize);
    }
    //WRITE OUTPUT BATCH
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime();
    {
      extern int *numpixs;
      extern int *pixstart;
      extern long *objglobalind;
      MPI_Request recvrequest[numproc];
      copyD2H_kernel(obj_h,obj_d,mynumpix*batchsize);
      #pragma omp parallel for
      for(int n = 0; n < mynumpix*batchsize; n++)
        pixsendbuff[n] = obj_h[n];
      if(myid == 0)
        for(int p = 0; p < numproc; p++)
          MPI_Irecv(pixrecvbuff+(long)pixstart[p]*batchsize,numpixs[p]*batchsize,MPI_FLOAT,p,0,MPI_COMM_WORLD,recvrequest+p);
      MPI_Ssend(pixsendbuff,mynumpix*batchsize,MPI_FLOAT,0,0,MPI_COMM_WORLD);
      if(myid == 0){
        MPI_Waitall(numproc,recvrequest,MPI_STATUSES_IGNORE);
        #pragma omp parallel for
        for(long n = 0; n < (long)numx*numy*batchsize; n++)
          objwritebuff[n] = pixrecvbuff[objglobalind[n]];
        double writetime = MPI_Wtime();
        fwrite(objwritebuff,sizeof(float),(long)numx*numy*batchsize,outputf);
        writetime = MPI_Wtime()-writetime;
        printf("WRITE TIME %e s (%f GB/s)\n",writetime,sizeof(float)*numx*numy*batchsize/writetime/1.0e9);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    iotime += MPI_Wtime()-time;
  }
  rtime = MPI_Wtime()-rtime;
  if(myid==0)fclose(inputf);
  if(myid==0)fclose(outputf);

  if(myid==0){
    double cgtime = rtime-ptime-btime-iotime;
    printf("\n");
    printf("recon: %e proj %e back %e cg %e i/o %e\n",rtime,ptime,btime,cgtime,iotime);
  }
  MPI_Allreduce(MPI_IN_PLACE,&ptime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&pktime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&pcstime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&pcntime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&pcrtime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&pchtime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&pmtime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&btime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&bktime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&bcstime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&bcntime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&bcrtime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&bchtime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&bmtime,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  if(myid==0){
    extern long proj_rownzall;
    extern long proj_warpnzall;
    extern long proj_mapnzall;
    extern long back_warpnzall;
    extern long back_mapnzall;
    extern long raynumoutall;
    extern long socketrayoutall;
    extern long noderayoutall;
    double pother = ptime-prtime-pktime-pmtime-pcstime-pcntime-pcrtime-pchtime;
    double bother = btime-brtime-bktime-bmtime-bcstime-bcntime-bcrtime-bchtime;
    printf("AGGREGATE proj %e ( %e %e %e %e %e %e %e ) back %e ( %e %e %e %e %e %e %e )\n",ptime,pktime,pcstime,pcntime,pchtime,pmtime,prtime,pother,btime,bktime,bcstime,bcntime,bchtime,bmtime,brtime,bother);
    printf("AGGREGATE total %e ( %e %e %e %e %e %e %e )\n",ptime+btime,pktime+bktime,pcstime+bcstime,pcntime+bcntime,pchtime+bchtime,pmtime+bmtime,prtime+brtime,pother+bother);
    printf("NUMBER OF PROJECTIONS %d BACKPROJECTIONS %d\n",numproj,numback);
    double projflop = proj_rownzall/1.0e9*2*(numiter)*numslice;
    double backflop = proj_rownzall/1.0e9*2*(numiter+1)*numslice;
    double flop = projflop+backflop;
    double projflopreal = (proj_warpnzall*WARPSIZE)/1.0e9*2*numproj*FFACTOR;
    double backflopreal = (back_warpnzall*WARPSIZE)/1.0e9*2*numback*FFACTOR;
    double flopreal = projflopreal+backflopreal;
    double projflops = projflop/pktime*numproc;
    double backflops = backflop/bktime*numproc;
    double flops = flop/(pktime+bktime)*numproc;
    double projflopsreal = projflopreal/pktime*numproc;
    double backflopsreal = backflopreal/bktime*numproc;
    double flopsreal = flopreal/(pktime+bktime)*numproc;

    double projshared = ((double)proj_warpnzall*WARPSIZE+proj_mapnzall)*sizeof(VECPREC)/1.0e9*numproj*FFACTOR;
    double backshared = ((double)back_warpnzall*WARPSIZE+back_mapnzall)*sizeof(VECPREC)/1.0e9*numback*FFACTOR;
    double shared = projshared+backshared;
    double projsharedbw = projshared/pktime*numproc;
    double backsharedbw = backshared/bktime*numproc;
    double sharedbw = shared/(pktime+bktime)*numproc;

    double projalgintshared = projflopreal/projshared;
    double backalgintshared = backflopreal/backshared;
    double aggalgintshared = flopreal/shared;

    double projglobal = (((double)proj_warpnzall*WARPSIZE*(sizeof(MATPREC)+sizeof(unsigned short))+proj_mapnzall*sizeof(int))+FFACTOR*sizeof(VECPREC)*((double)proj_mapnzall+raynumoutall))/1.0e9*numproj;
    double backglobal = (((double)back_warpnzall*WARPSIZE*(sizeof(MATPREC)+sizeof(unsigned short))+back_mapnzall*sizeof(int))+FFACTOR*sizeof(VECPREC)*((double)back_mapnzall+numpix))/1.0e9*numback;

    double global = projglobal+backglobal;
    double projglobalbw = projglobal/pktime*numproc;
    double backglobalbw = backglobal/bktime*numproc;
    double globalbw = (projglobal+backglobal)/(pktime+bktime)*numproc;

    double projalgintglobal = projflopreal/projglobal;
    double backalgintglobal = backflopreal/backglobal;
    double aggalgintglobal = flopreal/global;

    printf("\n");
    printf("ALGINTENSITY pkernel %f bkernel %f agg %f GLOBAL\n",projalgintglobal,backalgintglobal,aggalgintglobal);
    printf("ALGINTENSITY pkernel %f bkernel %f agg %f SHARED\n",projalgintshared,backalgintshared,aggalgintshared);
    printf("\n");
    printf("AGGREGATE pkernel %f (%f) bkernel %f (%f) agg %f (%f) TFLOPs\n",projflopreal/1.0e3,projflop/1.0e3,backflopreal/1.0e3,backflop/1.0e3,flopreal/1.0e3,flop/1.0e3);
    printf("AGGREGATE pkernel %f bkernel %f agg %f TB GLOBAL\n",projglobal/1.0e3,backglobal/1.0e3,global/1.0e3);
    printf("AGGREGATE pkernel %f bkernel %f agg %f TB SHARED\n",projshared/1.0e3,backshared/1.0e3,shared/1.0e3);
    printf("\n");
    printf("PERGPU pkernel %f (%f) bkernel %f (%f) agg %f (%f) GFLOPs\n",projflopreal/numproc,projflop/numproc,backflopreal/numproc,backflop/numproc,flopreal/numproc,flop/numproc);
    printf("PERGPU pkernel %f bkernel %f agg %f GB GLOBAL\n",projglobal/numproc,backglobal/numproc,global/numproc);
    printf("PERGPU pkernel %f bkernel %f agg %f GB SHARED\n",projshared/numproc,backshared/numproc,shared/numproc);
    printf("\n");
    printf("AGGREGATE pkernel %f (%f) bkernel %f (%f) agg %f (%f) TFLOPS\n",projflopsreal/1.0e3,projflops/1.0e3,backflopsreal/1.0e3,backflops/1.0e3,flopsreal/1.0e3,flops/1.0e3);
    printf("AGGREGATE pkernel %f bkernel %f agg %f TB/s GLOBAL\n",projglobalbw/1.0e3,backglobalbw/1.0e3,globalbw/1.0e3);
    printf("AGGREGATE pkernel %f bkernel %f agg %f TB/s SHARED\n",projsharedbw/1.0e3,backsharedbw/1.0e3,sharedbw/1.0e3);
    printf("\n");
    printf("PERGPU pkernel %f (%f) bkernel %f (%f) agg %f (%f) GFLOPS\n",projflopsreal/numproc,projflops/numproc,backflopsreal/numproc,backflops/numproc,flopsreal/numproc,flops/numproc);
    printf("PERGPU pkernel %f bkernel %f agg %f GB/s GLOBAL\n",projglobalbw/numproc,backglobalbw/numproc,globalbw/numproc);
    printf("PERGPU pkernel %f bkernel %f agg %f GB/s SHARED\n",projsharedbw/numproc,backsharedbw/numproc,sharedbw/numproc);
    printf("\n");
    extern long proj_intersocket;
    extern long proj_internode;
    extern long proj_interhost;
    extern long back_intersocket;
    extern long back_internode;
    extern long back_interhost;
    double socketdata = 2.0*raynumoutall/1.0e9*sizeof(COMMPREC)*(numproj+numback)*FFACTOR;
    double nodedata = 2.0*socketrayoutall/1.0e9*sizeof(COMMPREC)*(numproj+numback)*FFACTOR;
    double hostdata = 2.0*noderayoutall/1.0e9*sizeof(COMMPREC)*(numproj+numback)*FFACTOR;
    double socketdatareal = (proj_intersocket+back_intersocket)/1.0e9*sizeof(COMMPREC)*(numproj+numback)*FFACTOR;
    double nodedatareal = (proj_internode+back_internode)/1.0e9*sizeof(COMMPREC)*(numproj+numback)*FFACTOR;
    double hostdatareal = (proj_interhost+back_interhost)/1.0e9*sizeof(COMMPREC)*(numproj+numback)*FFACTOR;
    double socketbw = socketdata/(pcstime+bcstime)*numproc;
    double nodebw = nodedata/(pcntime+bcntime)*numproc;
    double hostbw = hostdata/(pchtime+bchtime)*numproc;
    double memcpybw = hostdata/(pmtime+bmtime)*numproc;
    double socketbwreal = socketdatareal/(pcstime+bcstime)*numproc;
    double nodebwreal = nodedatareal/(pcntime+bcntime)*numproc;
    double hostbwreal = hostdatareal/(pchtime+bchtime)*numproc;
    printf("AGGREGATE MEMCPY %f SOCKET %f (%f) NODE %f (%f) HOST %f (%f) GB\n",hostdata,socketdata,socketdatareal,nodedata,nodedatareal,hostdata,hostdatareal);
    printf("PERNODE MEMCPY %f SOCKET %f (%f) NODE %f (%f) HOST %f (%f) GB\n",hostdata/numnode,socketdata/numnode,socketdatareal/numnode,nodedata/numnode,nodedatareal/numnode,hostdata/numnode,hostdatareal/numnode);
    printf("PERSCKT MEMCPY %f SOCKET %f (%f) NODE %f (%f) HOST %f (%f) GB\n",hostdata/numsocket,socketdata/numsocket,socketdatareal/numsocket,nodedata/numsocket,nodedatareal/numsocket,hostdata/numsocket,hostdatareal/numsocket);
    printf("PERGPU MEMCPY %f SOCKET %f (%f) NODE %f (%f) HOST %f (%f) GB\n",hostdata/numproc,socketdata/numproc,socketdatareal/numproc,nodedata/numproc,nodedatareal/numproc,hostdata/numproc,hostdatareal/numproc);
    printf("\n");
    printf("AGGREGATE MEMCPY %f SOCKET %f (%f) NODE %f (%f) HOST %f (%f) GB/s\n",memcpybw,socketbw,socketbwreal,nodebw,nodebwreal,hostbw,hostbwreal);
    printf("PERNODE MEMCPY %f SOCKET %f (%f) NODE %f (%f) HOST %f (%f) GB/s\n",memcpybw/numnode,socketbw/numnode,socketbwreal/numnode,nodebw/numnode,nodebwreal/numnode,hostbw/numnode,hostbwreal/numnode);
    printf("PERSCKT MEMCPY %f SOCKET %f (%f) NODE %f (%f) HOST %f (%f) GB/s\n",memcpybw/numsocket,socketbw/numsocket,socketbwreal/numsocket,nodebw/numsocket,nodebwreal/numsocket,hostbw/numsocket,hostbwreal/numsocket);
    printf("PERGPU MEMCPY %f SOCKET %f (%f) NODE %f (%f) HOST %f (%f) GB/s\n",memcpybw/numproc,socketbw/numproc,socketbwreal/numproc,nodebw/numproc,nodebwreal/numproc,hostbw/numproc,hostbwreal/numproc);
    printf("\n");
  }

  /*float *mesall = new float[numray*FFACTOR];
  #pragma omp parallel for
  for(int n = 0; n < numray*FFACTOR; n++)
    mesall[n] = 0.0;
  for(int slice = 0; slice < FFACTOR; slice++)
    #pragma omp parallel for
    for(int k = 0; k < mynumray; k++)
      mesall[slice*numray+rayglobalind[k]] = mes_h[slice*mynumray+k];
  MPI_Allreduce(MPI_IN_PLACE,mesall,numray*FFACTOR,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
  if(myid==0){
    FILE *mesf = fopen("/gpfs/alpine/scratch/merth/csc362/mesall_new.bin","wb");
    fwrite(mesall,sizeof(float),numray*FFACTOR,mesf);
    fclose(mesf);
  }
  delete[] mesall;*/
  /*float *objall = new float[numpix*FFACTOR];
  #pragma omp parallel for
  for(int n = 0; n < numpix*FFACTOR; n++)
    objall[n] = 0.0;
  for(int slice = 0; slice < FFACTOR; slice++)
    #pragma omp parallel for
    for(int n = 0; n < mynumpix; n++)
      objall[slice*numpix+pixglobalind[n]] = obj_h[slice*mynumpix+n];
  MPI_Reduce(MPI_IN_PLACE,objall,numpix*FFACTOR,MPI_FLOAT,MPI_SUM,0,MPI_COMM_WORLD);
  if(myid==0){
    FILE *objf = fopen("/gpfs/alpine/scratch/merth/csc362/object_new.bin","wb");
    fwrite(objall,sizeof(float),numpix*FFACTOR,objf);
    fclose(objf);
  }
  delete[] objall;*/
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("Total Time: %e\n",MPI_Wtime()-timetot);

  MPI_Finalize();

  return 0;
};

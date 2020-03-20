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
int batchsize; //SLICE PER BATCH
int numbatch; //NUMBER OF BATCHES

double xstart; //X START OF DOMAIN
double ystart; //Y START OF DOMAIN
double pixsize = 1.0; //PIXEL SIZE
double rhostart; //RHO START
double raylength; //RAYLENGTH

char *sinfile; //SINOGRAM FILE
char *thefile; //THETA FILE
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
long raynumoutall;
int mynumray;
int mynumpix;

int *raysendstart;
int *rayrecvstart;
int *raysendcount;
int *rayrecvcount;

int *rayraystart;
int *rayrayind;

int *rayrecvlist;

//INDEX MAPPINGS
int *rayglobalind;
int *pixglobalind;
int *raymesind;

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
  chartemp = getenv("NUMX");
  numx = atoi(chartemp);
  //chartemp = getenv("NUMY");
  numy = numx;//atoi(chartemp);
  chartemp = getenv("NUMSLICE");
  numslice = atoi(chartemp);
  chartemp = getenv("BATCHSIZE");
  batchsize = atoi(chartemp);

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
    printf("                  BATCH SIZE: %d (%f + %f GB) %d FUSE FACTOR\n",batchsize,numr*numt*batchsize*sizeof(VECPREC)/1024.0/1024.0/1024.0,numx*numy*batchsize*sizeof(VECPREC)/1024.0/1024.0/1024.0,FFACTOR);
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
    printf("            BUFFER SIZE : %d (%f KB) / %d (%f KB)\n",proj_buffsize,proj_buffsize*(int)sizeof(VECPREC)/1024.0,back_buffsize,back_buffsize*(int)sizeof(VECPREC)/1024.0);
  }
  proj_buffsize = proj_buffsize/FFACTOR;
  back_buffsize = back_buffsize/FFACTOR;
  if(myid==0){
    printf("  EFFECTIVE BUFFER SIZE : %d (%f KB) / %d (%f KB)\n",proj_buffsize*FFACTOR,proj_buffsize*(int)sizeof(VECPREC)/1024.0*FFACTOR,back_buffsize*FFACTOR,back_buffsize*(int)sizeof(VECPREC)/1024.0*FFACTOR);
    printf("   BUFFER SIZE PER SLICE: %d (%f KB) / %d (%f KB)\n",proj_buffsize,proj_buffsize*(int)sizeof(VECPREC)/1024.0,back_buffsize,back_buffsize*(int)sizeof(VECPREC)/1024.0);
    printf("\n");
    printf("INTEGER: %d, DOUBLE: %d, LONG: %d, SHORT: %d, POINTER: %d\n",(int)sizeof(int),(int)sizeof(double),(int)sizeof(long),(int)sizeof(unsigned short),(int)sizeof(complex<double>*));
    printf("\n");
    printf("X & Y START   : (%f %f)\n",xstart,ystart);
    printf("RHO START     : %f\n",rhostart);
    printf("PIXEL SIZE    : %f\n",pixsize);
    printf("RAY LENGTH    : %f\n",raylength);
    printf("NUMBER OF ITERATIONS: %d\n",numiter);
    printf("SINOGRAM FILE : %s\n",sinfile);
    printf("   THETA FILE : %s\n",thefile);
    printf("\n");
    printf("NUMBER OF PROCS PER SOCKET: %d PER NODE: %d\n",numproc_socket,numproc_node);
    printf("NUMBER OF NODES: %d SOCKETS: %d PROCS: %d\n",numnode,numsocket,numproc);
    printf("\n");
  }

  double timep = MPI_Wtime();
  //PREPROCESSING
  preproc();
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("PREPROCESSING TIME: %e\n",MPI_Wtime()-timep);

  double *obj_h;//OBJECT
  double *gra_h;//GRADIENT
  double *dir_h;//DIRECTION
  double *mes_h;//MEASUREMENT
  double *ray_h;//RAYSUM
  double *res_h;//RESIDUE
  cudaMallocHost((void**)&obj_h,sizeof(double)*mynumpix*batchsize);
  cudaMallocHost((void**)&gra_h,sizeof(double)*mynumpix*batchsize);
  cudaMallocHost((void**)&dir_h,sizeof(double)*mynumpix*batchsize);
  cudaMallocHost((void**)&mes_h,sizeof(double)*mynumray*batchsize);
  cudaMallocHost((void**)&res_h,sizeof(double)*mynumray*batchsize);
  cudaMallocHost((void**)&ray_h,sizeof(double)*mynumray*batchsize);

  double *obj_d;//OBJECT
  double *gra_d;//GRADIENT
  double *dir_d;//DIRECTION
  double *mes_d;//MEASUREMENT
  double *ray_d;//RAYSUM
  double *res_d;//RESIDUE

  setup_gpu(&obj_d,&gra_d,&dir_d,&mes_d,&res_d,&ray_d);

  
  float *mesdata = new float[numt*numr*batchsize];
  float *recdata  = new float[numpix*batchsize];
  char recfile[1000];
  sprintf(recfile,"%s_rec",sinfile);
  if(myid==0)printf("RECONSTRUCTION FILE: %s\n",recfile);
  FILE *inputf;
  FILE *outputf;
  if(myid==0)inputf = fopen(sinfile,"rb");
  if(myid==0)outputf = fopen(recfile,"wb");
  if(myid==0)printf("CONJUGATE-GRADIENT OPTIMIZATION\n");
  MPI_Barrier(MPI_COMM_WORLD);
  double time = 0;
  rtime = MPI_Wtime();
  for(int batch = 0; batch < numbatch; batch++){
    if(myid==0)printf("BATCH %d\n",batch);
    //READ BATCH
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime();
    {
      if(myid==0)fread(mesdata,sizeof(float),numr*numt*batchsize,inputf);
      MPI_Bcast(mesdata,numr*numt*batchsize,MPI_FLOAT,0,MPI_COMM_WORLD);
      for(int slice = 0; slice < batchsize; slice++)
        #pragma omp parallel for
        for(int k = 0; k < mynumray; k++)
          if(raymesind[k]>-1)mes_h[slice*mynumray+k] = mesdata[slice*numt*numr+raymesind[k]];
          else mes_h[slice*mynumray+k] = 0;
      cudaMemcpy(mes_d,mes_h,sizeof(double)*mynumray*batchsize,cudaMemcpyHostToDevice);
      cudaMemset(obj_d,0,sizeof(double)*mynumpix*batchsize);
      //NORMALIZE
      extern int proj_maxnz;
      extern int back_maxnz;
      double mesmax = max_kernel(mes_h,mynumray*batchsize);
      double raymax = mesmax*back_maxnz*sqrt(2)*numx*sqrt(2);
      double scalefactor = 64e3/raymax;
      if(myid==0)printf("maximum possible: %e scale factor: %e\n",raymax,scalefactor);
      scale_kernel(mes_d,scalefactor,mynumray*batchsize);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    iotime += MPI_Wtime()-time;
    //FIND GRADIENT
    backproject(gra_d,mes_d);
    cudaMemcpy(gra_h,gra_d,sizeof(double)*mynumpix*batchsize,cudaMemcpyDeviceToHost);
    double resnorm = norm_kernel(mes_h,mynumray*batchsize);
    double resmax = max_kernel(mes_h,mynumray*batchsize);
    double gradnorm = norm_kernel(gra_h,mynumpix*batchsize);
    double gradmax = max_kernel(gra_h,mynumpix*batchsize);
    double objnorm = 0.0;
    double objmax = 0.0;
    if(myid==0)printf("iter: %d resnorm: %e resmax: %e gradnorm: %e gradmax: %e objnorm: %e objmax: %e\n",0,resnorm,resmax,gradnorm,gradmax,objnorm,objmax);
    //SAVE DIRECTION
    double oldgradnorm = gradnorm;
    copy_kernel(dir_d,gra_d,mynumpix*batchsize);
    copy_kernel(res_d,mes_d,mynumray*batchsize);
    //START ITERATIONS
    for(int iter = 1; iter <= numiter; iter++){
      //PROJECT DIRECTION
      projection(ray_d,dir_d);
      cudaMemcpy(ray_h,ray_d,sizeof(double)*mynumray*batchsize,cudaMemcpyDeviceToHost);
      cudaMemcpy(res_h,res_d,sizeof(double)*mynumray*batchsize,cudaMemcpyDeviceToHost);
      //FIND STEP SIZE
      double temp1 = dot_kernel(res_h,ray_h,mynumray*batchsize);
      double temp2 = norm_kernel(ray_h,mynumray*batchsize);
      //STEP SIZE
      double alpha = temp1/temp2;
      saxpy_kernel(obj_d,obj_d,alpha,dir_d,mynumpix*batchsize);
      //FORWARD PROJECTION
      projection(ray_d,obj_d);
      //FIND RESIDUAL ERROR
      saxpy_kernel(res_d,mes_d,-1.0,ray_d,mynumray*batchsize);
      //FIND GRADIENT
      backproject(gra_d,res_d);
      cudaMemcpy(res_h,res_d,sizeof(double)*mynumray*batchsize,cudaMemcpyDeviceToHost);
      cudaMemcpy(gra_h,gra_d,sizeof(double)*mynumpix*batchsize,cudaMemcpyDeviceToHost);
      cudaMemcpy(obj_h,obj_d,sizeof(double)*mynumpix*batchsize,cudaMemcpyDeviceToHost);
      resnorm = norm_kernel(res_h,mynumray*batchsize);
      resmax = max_kernel(res_h,mynumray*batchsize);
      gradnorm = norm_kernel(gra_h,mynumpix*batchsize);
      gradmax = max_kernel(gra_h,mynumpix*batchsize);
      objnorm = norm_kernel(obj_h,mynumpix*batchsize);
      objmax = max_kernel(obj_h,mynumpix*batchsize);
      if(myid==0)printf("iter: %d resnorm: %e resmax: %e gradnorm: %e gradmax: %e objnorm: %e objmax: %e\n",iter,resnorm,resmax,gradnorm,gradmax,objnorm,objmax);
      //UPDATE DIRECTION
      double beta = gradnorm/oldgradnorm;
      //double beta = 0;
      oldgradnorm = gradnorm;
      saxpy_kernel(dir_d,gra_d,beta,dir_d,mynumpix*batchsize);
    }
    //WRITE SLICE
    MPI_Barrier(MPI_COMM_WORLD);
    time = MPI_Wtime();
    {
      /*#pragma omp parallel for
      for(int n = 0; n < numpix*batchsize; n++)
        recdata[n] = 0;
      for(int slice = 0; slice < batchsize; slice++)
        #pragma omp parallel for
        for(int n = 0; n < mynumpix; n++)
          recdata[slice*numpix+pixglobalind[n]] = obj_h[slice*mynumpix+n];
      if(myid==0)
        MPI_Reduce(MPI_IN_PLACE,recdata,numpix*batchsize,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      else
        MPI_Reduce(recdata,recdata,numpix*batchsize,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      if(myid==0)fwrite(recdata,sizeof(double),numpix*batchsize,recf);*/
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
    extern int proj_mapnzall;
    extern long back_warpnzall;
    extern int back_mapnzall;
    extern long socketrayoutall;
    extern long noderayoutall;
    double pother = ptime-prtime-pktime-pmtime-pcstime-pcntime-pcrtime-pchtime;
    double bother = btime-brtime-bktime-bmtime-bcstime-bcntime-bcrtime-bchtime;
    printf("AGGREGATE proj %e ( %e %e %e %e %e %e %e ) back %e ( %e %e %e %e %e %e %e )\n",ptime,pktime,pmtime,pcstime,pcntime,pchtime,prtime,pother,btime,bktime,bmtime,bcstime,bcntime,bchtime,prtime,bother);
    printf("NUMBER OF PROJECTIONS %d BACKPROJECTIONS %d\n",numproj,numback);
    double aggprojflop = proj_rownzall/1.0e9*2*(2*numiter)*numslice;
    double aggbackflop = proj_rownzall/1.0e9*2*(numiter+1)*numslice;
    double aggflop = aggprojflop+aggbackflop;
    double aggprojflops = aggprojflop/pktime*numproc;
    double aggbackflops = aggbackflop/bktime*numproc;
    double aggflops = aggflop/(pktime+bktime)*numproc;

    double aggprojshared = ((double)proj_warpnzall*WARPSIZE+proj_mapnzall)*sizeof(VECPREC)/1024.0/1024.0/1024.0*numproj*FFACTOR;
    double aggbackshared = ((double)back_warpnzall*WARPSIZE+back_mapnzall)*sizeof(VECPREC)/1024.0/1024.0/1024.0*numback*FFACTOR;
    double aggshared = aggprojshared+aggbackshared;
    double aggprojsharedbw = aggprojshared/pktime*numproc;
    double aggbacksharedbw = aggbackshared/bktime*numproc;
    double aggsharedbw = aggshared/(pktime+bktime)*numproc;

    double aggprojglobal = (((double)proj_warpnzall*WARPSIZE*(sizeof(MATPREC)+sizeof(unsigned short))+proj_mapnzall*sizeof(int))+(double)FFACTOR*(proj_mapnzall*sizeof(VECPREC)+raynumoutall*(sizeof(VECPREC)+sizeof(int))+numray*sizeof(double)))/1024.0/1024.0/1024.0*numproj;
    double aggbackglobal = (((double)back_warpnzall*WARPSIZE*(sizeof(MATPREC)+sizeof(unsigned short))+back_mapnzall*sizeof(int))+(double)FFACTOR*(back_mapnzall*sizeof(VECPREC)+noderayoutall*(sizeof(VECPREC)+sizeof(int))+numpix*sizeof(double)))/1024.0/1024.0/1024.0*numback;
    double aggglobal = aggprojglobal+aggbackglobal;
    double aggprojglobalbw = aggprojglobal/pktime*numproc;
    double aggbackglobalbw = aggbackglobal/bktime*numproc;
    double aggglobalbw = (aggprojglobal+aggbackglobal)/(pktime+bktime)*numproc;
    printf("\n");
    printf("AGGREGATE pkernel %f bkernel %f agg %f TFLOPs\n",aggprojflop/1.0e3,aggbackflop/1.0e3,aggflop/1.0e3);
    printf("AGGREGATE pkernel %f bkernel %f agg %f TB GLOBAL\n",aggprojglobal/1024.0,aggbackglobal/1024.0,aggglobal/1024.0);
    printf("AGGREGATE pkernel %f bkernel %f agg %f TB SHARED\n",aggprojshared/1024.0,aggbackshared/1024.0,aggshared/1024.0);
    printf("\n");
    //printf("PERGPU pkernel %f bkernel %f agg %f GFLOPS\n",aggprojflop/numproc,aggbackflop/numproc,aggflop/numproc);
    //printf("PERGPU pkernel %f bkernel %f agg %f GB/s GLOBAL\n",aggprojglobal/numproc,aggbackglobal/numproc,aggglobal/numproc);
    //printf("PERGPU pkernel %f bkernel %f agg %f GB/s SHARED\n",aggprojshared/numproc,aggbackshared/numproc,aggshared/numproc);
    //printf("\n");
    printf("AGGREGATE pkernel %f bkernel %f agg %f TFLOPS\n",aggprojflops/1.0e3,aggbackflops/1.0e3,aggflops/1.0e3);
    printf("AGGREGATE pkernel %f bkernel %f agg %f TB/s GLOBAL\n",aggprojglobalbw/1024.0,aggbackglobalbw/1024.0,aggglobalbw/1024.0);
    printf("AGGREGATE pkernel %f bkernel %f agg %f TB/s SHARED\n",aggprojsharedbw/1024.0,aggbacksharedbw/1024.0,aggsharedbw/1024.0);
    printf("\n");
    printf("PERGPU pkernel %f bkernel %f agg %f GFLOPS\n",aggprojflops/numproc,aggbackflops/numproc,aggflops/numproc);
    printf("PERGPU pkernel %f bkernel %f agg %f GB/s GLOBAL\n",aggprojglobalbw/numproc,aggbackglobalbw/numproc,aggglobalbw/numproc);
    printf("PERGPU pkernel %f bkernel %f agg %f GB/s SHARED\n",aggprojsharedbw/numproc,aggbacksharedbw/numproc,aggsharedbw/numproc);
    printf("\n");
    double socketdata = 2.0*raynumoutall*sizeof(VECPREC)*(numproj+numback)/1024.0/1024.0/1024.0*FFACTOR;
    double nodedata = 2.0*socketrayoutall*sizeof(VECPREC)*(numproj+numback)/1024.0/1024.0/1024.0*FFACTOR;
    double hostdata = 2.0*noderayoutall*sizeof(VECPREC)*(numproj+numback)/1024.0/1024.0/1024.0*FFACTOR;
    double socketbw = socketdata/(pcstime+bcstime)*numproc;
    double nodebw = nodedata/(pcntime+bcntime)*numproc;
    double hostbw = hostdata/(pchtime+bchtime)*numproc;
    double memcpybw = hostdata/(pmtime+bmtime)*numproc;
    printf("AGGREGATE MEMCPY %f GB SOCKET %f GB NODE %f GB HOST %f GB\n",hostdata,socketdata,nodedata,hostdata);
    printf("PERNODE MEMCPY %f GB SOCKET %f GB NODE %f GB HOST %f GB\n",hostdata/numnode,socketdata/numnode,nodedata/numnode,hostdata/numnode);
    printf("PERSCKT MEMCPY %f GB SOCKET %f GB NODE %f GB HOST %f GB\n",hostdata/numsocket,socketdata/numsocket,nodedata/numsocket,hostdata/numsocket);
    printf("PERGPU MEMCPY %f GB SOCKET %f GB NODE %f GB HOST %f GB\n",hostdata/numproc,socketdata/numproc,nodedata/numproc,hostdata/numproc);
    printf("\n");
    printf("AGGREGATE MEMCPY %f GB/s SOCKET %f GB/s NODE %f GB/s HOST %f GB/s\n",memcpybw,socketbw,nodebw,hostbw);
    printf("PERNODE MEMCPY %f GB/s SOCKET %f GB/s NODE %f GB/s HOST %f GB/s\n",memcpybw/numnode,socketbw/numnode,nodebw/numnode,hostbw/numnode);
    printf("PERSCKT MEMCPY %f GB/s SOCKET %f GB/s NODE %f GB/s HOST %f GB/s\n",memcpybw/numsocket,socketbw/numsocket,nodebw/numsocket,hostbw/numsocket);
    printf("PERGPU MEMCPY %f GB/s SOCKET %f GB/s NODE %f GB/s HOST %f GB/s\n",memcpybw/numproc,socketbw/numproc,nodebw/numproc,hostbw/numproc);
    printf("\n");
  }
  float *mesall = new float[numray];
  #pragma omp parallel for
  for(int n = 0; n < numray; n++)
    mesall[n] = 0;
  #pragma omp parallel for
  for(int k = 0; k < mynumray; k++)
    mesall[rayglobalind[k]] = mes_h[k];
  MPI_Allreduce(MPI_IN_PLACE,mesall,numray,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
  if(myid==0){
    FILE *mesf = fopen("/gpfs/alpine/scratch/merth/csc362/mesall_new.bin","wb");
    fwrite(mesall,sizeof(float),numray,mesf);
    fclose(mesf);
  }
  delete[] mesall;
  float *objall = new float[numpix];
  #pragma omp parallel for
  for(int n = 0; n < numpix; n++)
    objall[n] = 0.0;
  #pragma omp parallel for
  for(int n = 0; n < mynumpix; n++)
    objall[pixglobalind[n]] = obj_h[n];
  MPI_Allreduce(MPI_IN_PLACE,objall,numpix,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
  if(myid==0){
    FILE *objf = fopen("/gpfs/alpine/scratch/merth/csc362/object_new.bin","wb");
    fwrite(objall,sizeof(float),numpix,objf);
    fclose(objf);
  }
  delete[] objall;
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("Total Time: %e\n",MPI_Wtime()-timetot);

  MPI_Finalize();

  return 0;
}

double norm_kernel(double *a, int dim){
  double reduce = 0;
  #pragma omp parallel for reduction(+:reduce)
  for(int n = 0; n < dim; n++)
    reduce += norm(a[n]);
  MPI_Allreduce(MPI_IN_PLACE,&reduce,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  return reduce;
};
double max_kernel(double *a, int dim){
  double reduce = 0;
  #pragma omp parallel for reduction(max:reduce)
  for(int n = 0; n < dim; n++){
    double test = abs(a[n]);
    if(test>reduce)reduce=test;
  }
  MPI_Allreduce(MPI_IN_PLACE,&reduce,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  return reduce;
};
double dot_kernel(double *a, double *b, int dim){
  double reduce = 0;
  #pragma omp parallel for reduction(+:reduce)
  for(int n = 0; n < dim; n++)
    reduce += a[n]*b[n];
  MPI_Allreduce(MPI_IN_PLACE,&reduce,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  return reduce;
};

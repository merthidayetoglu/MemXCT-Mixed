#include "vars.h"

extern int numx; //X SIZE OF DOMAIN
extern int numy; //Y SIZE OF DOMAIN
extern int numt; //NUMBER OF THETAS
extern int numr; //NUMBER OF RHOS

extern double xstart; //X START OF DOMAIN
extern double ystart; //Y START OF DOMAIN
extern double pixsize; //PIXEL SIZE
extern double rhostart; //RHO START
extern double raylength; //RAYLENGTH

extern char *sinfile;
extern int numiter;

extern int spatsize; //SPATIAL TILE SIZE
extern int specsize; //SPECTRAL TILE SIZE
extern int numxtile; //NUMBER OF X TILES
extern int numytile; //NUMBER OF Y TILES
extern int numttile; //NUMBER OF THETA TILES
extern int numrtile; //NUMBER OF RHO TILES
extern int numpix; //NUMBER OF PIXELS (EXTENDED)
extern int numray; //NUMBER OF RAYS (EXTENDED)
extern int numspattile; //NUMBER OF SPATIAL TILES
extern int numspectile; //NUMBER OF SPECTRAL TILES

long raynumoutall;
extern int raynuminc;
extern int raynumout;
extern int mynumray;
extern int mynumpix;

int *raysendstart;
int *rayrecvstart;
int *raysendcount;
int *rayrecvcount;

int *rayraystart;
int *rayrayind;

extern int proj_blocksize;
extern int proj_buffsize;
extern int back_blocksize;
extern int back_buffsize;

complex<double> *pixcoor;

double proj_rowmax;
long proj_rownztot;
long proj_rownzall;
long *proj_rowdispl;
int *proj_rowindex;
int proj_numblocks;
int proj_numbufftot;
int proj_numbuffall;
int *proj_buffdispl;
int proj_mapnztot;
long proj_mapnzall;
int *proj_mapdispl;
int *proj_mapnz;
int *proj_buffmap;
int proj_warpnztot;
long proj_warpnzall;
int *proj_warpdispl;
bool *proj_warpindextag;

double back_rowmax;
long *back_rowdispl;
int *back_rowindex;
int back_numblocks;
int back_numbufftot;
int back_numbuffall;
int *back_buffdispl;
int back_mapnztot;
long back_mapnzall;
int *back_mapdispl;
int *back_mapnz;
int *back_buffmap;
int back_warpnztot;
long back_warpnzall;
int *back_warpdispl;
bool *back_warpindextag;
#ifdef MATRIX
matrix *proj_warpindval;
matrix *back_warpindval;
#else
unsigned short *proj_warpindex;
MATPREC *proj_warpvalue;
unsigned short *back_warpindex;
MATPREC *back_warpvalue;
#endif

int *rayglobalind;
int *pixglobalind;
int *raymesind;
int *pixobjind;
long *objglobalind;
long *mesglobalind;

int *numpixs;
int *numrays;
int *pixstart;
int *raystart;

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

void preproc(){

  MPI_Barrier(MPI_COMM_WORLD);
  double time = MPI_Wtime();
  if(myid==0)printf("PLACE TILES\n");
  //PLACE SPATIAL TILES
  int lspat = numxtile;
  if(numytile > lspat)lspat = numytile;
  int spatlevel = 0;
  while(true){
    if(lspat<=pow(2,spatlevel))break;
    spatlevel++;
  }
  int lspatdim = pow(2,spatlevel);
  complex<double> *spatlltemp = new complex<double>[lspatdim*lspatdim];
  #pragma omp parallel for
  for(int lspat = 0; lspat < lspatdim*lspatdim; lspat++)
    spatlltemp[lspat].real(xstart-1);
  //#pragma omp parallel for
  for(int spat = 0; spat < numspattile; spat++){
    int ytile = spat/numxtile;
    int xtile = spat%numxtile;
    int ind = 0;
      //ind = ytile*numxtile+xtile;
      //ind = encode(xtile,ytile);
      ind = xy2d (lspatdim,xtile,ytile);
    double x = xstart+xtile*spatsize;
    double y = ystart+ytile*spatsize;
    spatlltemp[ind]=complex<double>(x,y);
  }
  complex<double> *spatll = new complex<double>[numspattile];
  int spatcount = 0;
  for(int lspat = 0; lspat < lspatdim*lspatdim; lspat++)
    if(spatlltemp[lspat].real()>xstart-0.5){
      spatll[spatcount] = spatlltemp[lspat];
      spatcount++;
    }
  delete[] spatlltemp;
  //PLACE SPECTRAL TILES
  int lspec = numrtile;
  if(numttile > lspec)lspec = numttile;
  int speclevel = 0;
  while(true){
    if(lspec<=pow(2,speclevel))break;
    speclevel++;
  }
  int lspecdim = pow(2,speclevel);
  complex<double> *speclltemp = new complex<double>[lspecdim*lspecdim];
  #pragma omp parallel for
  for(int lspec = 0; lspec < lspecdim*lspecdim; lspec++)
    speclltemp[lspec].real(rhostart-1.0);
  #pragma omp parallel for
  for(int spec = 0; spec < numspectile; spec++){
    int thetile = spec/numrtile;
    int rhotile = spec%numrtile;
    int ind = 0;
      //ind = thetile*numrtile+rhotile;
      //ind = encode(rhotile,thetile);
      ind = xy2d(lspecdim,rhotile,thetile);
    double rho = rhostart+rhotile*specsize;
    double the = thetile*specsize*M_PI/numt;
    speclltemp[ind]=complex<double>(rho,the);
  }
  complex<double> *specll = new complex<double>[numspectile];
  int speccount = 0;
  for(int lspec = 0; lspec < lspecdim*lspecdim; lspec++)
    if(speclltemp[lspec].real()>rhostart-0.5){
      specll[speccount] = speclltemp[lspec];
      speccount++;
    }
  delete[] speclltemp;
  if(myid==0)printf("MPI PARTITIONING\n");
  int numspats[numproc_data];
  int numspecs[numproc_data];
  int spatstart[numproc_data];
  int specstart[numproc_data];
  numpixs = new int[numproc_data];
  numrays = new int [numproc_data];
  pixstart = new int[numproc_data];
  raystart = new int[numproc_data];
  int myspattemp = (numspattile/numproc_data)*numproc_data;
  int myspectemp = (numspectile/numproc_data)*numproc_data;
  for(int p = 0; p < numproc_data; p++){
    numspats[p] = numspattile/numproc_data;
    numspecs[p] = numspectile/numproc_data;
    if(myspattemp < numspattile){
      numspats[p]++;
      myspattemp++;
    }
    if(myspectemp < numspectile){
      numspecs[p]++;
      myspectemp++;
    }
  }
  spatstart[0] = 0;
  specstart[0] = 0;
  for(int p = 1; p < numproc_data; p++){
    spatstart[p] = spatstart[p-1] + numspats[p-1];
    specstart[p] = specstart[p-1] + numspecs[p-1];
  }
  for(int p = 0; p < numproc_data; p++){
    numpixs[p] = numspats[p]*spatsize*spatsize;
    numrays[p] = numspecs[p]*specsize*specsize;
  }
  pixstart[0] = 0;
  raystart[0] = 0;
  for(int p = 1; p < numproc_data; p++){
    pixstart[p] = pixstart[p-1] + numpixs[p-1];
    raystart[p] = raystart[p-1] + numrays[p-1];
  }
  mynumpix = numpixs[myid_data];
  mynumray = numrays[myid_data];
  int maxnumpix = numpixs[0];
  int maxnumray = numrays[0];
  int minnumpix = numpixs[0];
  int minnumray = numrays[0];
  for(int p = 0; p < numproc_data; p++){
    if(numpixs[p]>maxnumpix)maxnumpix=numpixs[p];
    if(numrays[p]>maxnumray)maxnumray=numrays[p];
    if(numpixs[p]<minnumpix)minnumpix=numpixs[p];
    if(numrays[p]<minnumray)minnumray=numrays[p];
  }
  if(myid==0){
    for(int p = 0; p < numproc_data; p++)
      printf("proc: %d numspats: %d numpixs: %d (%d blocks) /  numspecs: %d numrays: %d (%d blocks)\n",p,numspats[p],numpixs[p],numpixs[p]/back_blocksize,numspecs[p],numrays[p],numrays[p]/proj_blocksize);
    printf("minnumpix: %d maxnumpix: %d imbalance: %f\n",minnumpix,maxnumpix,maxnumpix/((double)(numpix)/numproc_data));
    printf("minnumray: %d maxnumray: %d imbalance: %f\n",minnumray,maxnumray,maxnumray/((double)(numray)/numproc_data));
  }
  if(myid==0)printf("FILL PIXELS AND RAYS\n");
  //PLACE PIXELS
  pixcoor = new complex<double>[mynumpix];
  pixglobalind = new int[mynumpix];
  pixobjind = new int[mynumpix];
  //#pragma omp parallel for
  for(int pix = 0; pix < mynumpix; pix++){
    int tile = pix/(spatsize*spatsize);
    int pixloc = pix%(spatsize*spatsize);
    int pixlocy = pixloc/spatsize;
    int pixlocx = pixloc%spatsize;
    int  ind = tile*spatsize*spatsize + xy2d(spatsize,pixlocx,pixlocy);
    double x = spatll[spatstart[myid_data]+tile].real()+0.5+pixlocx;
    double y = spatll[spatstart[myid_data]+tile].imag()+0.5+pixlocy;
    pixcoor[ind] = complex<double>(x,y);
    //GLOBAL SPATIAL INDEX (EXTENDED)
    int xglobalind = (int)(x-xstart);
    int yglobalind = (int)(y-ystart);
    pixglobalind[ind] = yglobalind*numxtile*spatsize+xglobalind;
    if(xglobalind < numx && yglobalind < numy)
      pixobjind[ind] = yglobalind*numx+xglobalind;
    else
      pixobjind[ind] = -1;
  }
  float *mestheta = new float[numt];
  if(myid==0)printf("INPUT THETA DATA\n");
  extern char *thefile;
  FILE *thetaf = fopen(thefile,"rb");
  fread(mestheta,sizeof(float),numt,thetaf);
  /*if(myid==0)
    for(int n = 0; n < numt; n++)
      printf("%e degrees %d/%d\n",mestheta[n]/M_PI*180,n,numt);*/
  fclose(thetaf);
  //PLACE RAYS
  complex<double> *raycoor = new complex<double>[mynumray];
  rayglobalind = new int[mynumray];
  raymesind = new int[mynumray];
  #pragma omp parallel for
  for(int ray = 0; ray < mynumray; ray++){
    int tile = ray/(specsize*specsize);
    int rayloc = ray%(specsize*specsize);
    int raylocthe = rayloc/specsize;
    int raylocrho = rayloc%specsize;
    int ind = tile*specsize*specsize + xy2d(specsize,raylocrho,raylocthe);
    double rho = specll[specstart[myid_data]+tile].real()+0.5+raylocrho;
    double the = specll[specstart[myid_data]+tile].imag()+raylocthe*M_PI/numt;
    //GLOBAL SPECTRAL INDEX (EXTENDED)
    int rhoglobalind = (int)((rho-rhostart));
    int theglobalind = (int)((the+(M_PI/numt)/2)/(M_PI/numt));
    rayglobalind[ind] = theglobalind*numrtile*specsize+rhoglobalind;
    if(theglobalind < numt && rhoglobalind < numr){
      raymesind[ind] = theglobalind*numr+rhoglobalind;
      raycoor[ind] = complex<double>(rho,mestheta[theglobalind]);
      //raycoor[ind] = complex<double>(rho,the);
    }
    else{
      raycoor[ind].real(5*raylength);
      raymesind[ind] = -1;
    }
  }
  delete[] mestheta;
  delete[] specll;
  if(myid==0)printf("DOMAIN PARTITIONING\n");
  rayrecvcount = new int[numproc_data];
  raysendcount = new int[numproc_data];
  rayrecvstart = new int[numproc_data];
  raysendstart = new int[numproc_data];
  double *lengthtemp = new double[mynumray];
  int *rayrecvtemp[numproc_data];
  for(int p = 0; p < numproc_data; p++){
    #pragma omp parallel for
    for(int k = 0; k < mynumray; k++){
      lengthtemp[k] = 0;
      double rho = raycoor[k].real();
      double theta = raycoor[k].imag();
      for(int tile = spatstart[p]; tile < spatstart[p]+numspats[p]; tile++){
        double domain[4];
        domain[0]=spatll[tile].real();
        domain[1]=domain[0]+spatsize;
        domain[2]=spatll[tile].imag();
        domain[3]=domain[2]+spatsize;
        //REMOVE SPATIAL EDGE CONDITION
        if(domain[1] > xstart+numx)domain[1]=xstart+numx;
        if(domain[3] > ystart+numy)domain[3]=ystart+numy;
        findlength(theta,rho,&domain[0],&lengthtemp[k]);
      }
    }
    rayrecvcount[p] = 0;
    for(int k = 0; k < mynumray; k++)
      if(lengthtemp[k]>0)
        rayrecvcount[p]++;
    rayrecvtemp[p] = new int[rayrecvcount[p]];
    rayrecvcount[p] = 0;
    for(int k = 0; k < mynumray; k++)
      if(lengthtemp[k]>0){
        rayrecvtemp[p][rayrecvcount[p]]=k;
        rayrecvcount[p]++;
      }
  }
  delete[] lengthtemp;
  //EXCHANGE SEND & RECV MAPS
  MPI_Alltoall(rayrecvcount,1,MPI_INTEGER,raysendcount,1,MPI_INTEGER,MPI_COMM_DATA);
  rayrecvstart[0] = 0;
  raysendstart[0] = 0;
  for(int p = 1; p < numproc_data; p++){
    rayrecvstart[p] = rayrecvstart[p-1] + rayrecvcount[p-1];
    raysendstart[p] = raysendstart[p-1] + raysendcount[p-1];
  }
  raynuminc = rayrecvstart[numproc_data-1]+rayrecvcount[numproc_data-1];
  raynumout = raysendstart[numproc_data-1]+raysendcount[numproc_data-1];
  long raynumincall = raynuminc;
  raynumoutall = raynumout;
  MPI_Allreduce(MPI_IN_PLACE,&raynumincall,1,MPI_LONG,MPI_SUM,MPI_COMM_DATA);
  MPI_Allreduce(MPI_IN_PLACE,&raynumoutall,1,MPI_LONG,MPI_SUM,MPI_COMM_DATA);
  int raynumouts[numproc_data];
  int raynumincs[numproc_data];
  MPI_Allgather(&raynumout,1,MPI_INT,raynumouts,1,MPI_INT,MPI_COMM_DATA);
  MPI_Allgather(&raynuminc,1,MPI_INT,raynumincs,1,MPI_INT,MPI_COMM_DATA);
  int raynumoutmin = raynumouts[0];
  int raynumoutmax = raynumouts[0];
  int raynumincmin = raynumincs[0];
  int raynumincmax = raynumincs[0];
  for(int p = 0; p < numproc_data; p++){
    if(raynumoutmin>raynumouts[p])raynumoutmin=raynumouts[p];
    if(raynumoutmax<raynumouts[p])raynumoutmax=raynumouts[p];
    if(raynumincmin>raynumincs[p])raynumincmin=raynumincs[p];
    if(raynumincmax<raynumincs[p])raynumincmax=raynumincs[p];
  }
  if(myid==0){
    printf("total outgoing rays: %ld %fx (%f blocks av. %f per proc)\n",raynumoutall,raynumoutall/(double)(numt*numr),raynumoutall/(double)proj_blocksize,raynumoutall/(double)proj_blocksize/numproc_data);
    for(int p = 0; p < numproc_data; p++)
      printf("proc: %d raynumout: %d (%d blocks) raynuminc: %d (%d blocks)\n",p,raynumouts[p],raynumouts[p]/proj_blocksize,raynumincs[p],raynumincs[p]/back_blocksize);
    printf("raynumoutmin: %d raynumoutmax: %d imbalance: %f\n",raynumoutmin,raynumoutmax,raynumoutmax/((double)raynumoutall/numproc_data));
    printf("raynumincmin: %d raynumincmax: %d imbalance: %f\n",raynumincmin,raynumincmax,raynumincmax/((double)raynumincall/numproc_data));
  }
  int *raysendlist = new int[raynumout];
  int *rayrecvlist = new int[raynuminc];
  for(int p = 0; p < numproc_data; p++){
    #pragma omp parallel for
    for(int k = 0; k < rayrecvcount[p]; k++)
      rayrecvlist[rayrecvstart[p]+k] = rayrecvtemp[p][k];
    delete[] rayrecvtemp[p];
  }
  MPI_Alltoallv(rayrecvlist,rayrecvcount,rayrecvstart,MPI_INTEGER,raysendlist,raysendcount,raysendstart,MPI_INTEGER,MPI_COMM_DATA);
  //EXCHANGE RAY COORDINATES
  complex<double> *raycoorinc = new complex<double>[raynuminc];
  complex<double> *raycoorout = new complex<double>[raynumout];
  #pragma omp parallel for
  for(int k = 0; k < raynuminc; k++)
    raycoorinc[k] = raycoor[rayrecvlist[k]];
  MPI_Alltoallv(raycoorinc,rayrecvcount,rayrecvstart,MPI_DOUBLE_COMPLEX,raycoorout,raysendcount,raysendstart,MPI_DOUBLE_COMPLEX,MPI_COMM_DATA);
  delete[] raycoor;
  //FIND RAY-TO-RAY MAPPING
  int *raynumray = new int[mynumray];
  #pragma omp parallel for
  for(int k = 0; k < mynumray; k++)
    raynumray[k]=0;
  for(int k = 0; k < raynuminc; k++)
    raynumray[rayrecvlist[k]]++;
  rayraystart = new int[mynumray+1];
  rayraystart[0] = 0;
  for(int k = 1; k < mynumray+1; k++)
    rayraystart[k] = rayraystart[k-1] + raynumray[k-1];
  rayrayind = new int[raynuminc];
  #pragma omp parallel for
  for(int k = 0; k < mynumray; k++)raynumray[k]=0;
  for(int k = 0; k < raynuminc; k++){
    rayrayind[rayraystart[rayrecvlist[k]]+raynumray[rayrecvlist[k]]] = k;
    raynumray[rayrecvlist[k]]++;
  }
  delete[] raynumray;
  delete[] raysendlist;
  delete[] rayrecvlist;
  //OBJECT MAPPING
  {
    extern int iobatchsize;
    int *pixobjinds;
    if(myid_data == 0){
      pixobjinds = new int[numpix];
      objglobalind = new long[(long)numx*numy*iobatchsize];
      if(myid==0)printf("OBJECT OUTPUTMAP: %ld (%f GB)\n",(long)numx*numy*iobatchsize,sizeof(long)*numx*numy*iobatchsize/1.0e9);
    }
    MPI_Gatherv(pixobjind,mynumpix,MPI_INT,pixobjinds,numpixs,pixstart,MPI_INT,0,MPI_COMM_DATA);
    if(myid_data==0){
      for(int p = 0; p < numproc_data; p++)
        #pragma omp parallel for
        for(int n = 0; n < numpixs[p]; n++){
          int ind = pixobjinds[pixstart[p]+n];
          if(ind > -1)
            for(int slice = 0; slice < iobatchsize; slice++)
              objglobalind[(long)slice*numx*numy+ind] = (long)iobatchsize*pixstart[p]+(long)slice*numpixs[p]+n;
        }
      delete[] pixobjinds;
    }
    extern int iobatchsize;
    int *raymesinds;
    if(myid_data == 0){
      raymesinds = new int[numray];
      mesglobalind = new long[(long)numray*iobatchsize];
      if(myid==0)printf("MEASUREMENT INPUTMAP: %ld (%f GB)\n",(long)numray*iobatchsize,sizeof(long)*numray*iobatchsize/1.0e9);
    }
    MPI_Gatherv(raymesind,mynumray,MPI_INT,raymesinds,numrays,raystart,MPI_INT,0,MPI_COMM_DATA);
    if(myid_data == 0){
      for(int p = 0; p < numproc_data; p++)
        #pragma omp parallel for
        for(int n = 0; n < numrays[p]; n++){
          int ind = raymesinds[raystart[p]+n];
          if(ind > -1)
            for(int slice = 0; slice < iobatchsize; slice++)
              mesglobalind[(long)iobatchsize*raystart[p]+(long)slice*numrays[p]+n] = (long)slice*numr*numt+ind;
          else
            for(int slice = 0; slice < iobatchsize; slice++)
              mesglobalind[(long)iobatchsize*raystart[p]+(long)slice*numrays[p]+n] = -1;
        }
      delete[] raymesinds;
    }
  }
    /*{
      int N = 1000000;
      int *senddata = new int[N*numproc_data];
      int *recvdata = new int[N*numproc_data];
      {
        MPI_Barrier(MPI_COMM_WORLD);
        double time = MPI_Wtime();
        if(myid==0)MPI_Send(senddata,N,MPI_INT,6,0,MPI_COMM_WORLD);
        if(myid==6)MPI_Recv(recvdata,N,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);
        time = MPI_Wtime()-time;
        printf("inter-node time %e data %e GB B/W %e (GB/s)\n",time,N/1024.0/1024.0/1024.0*4,N/1024.0/1024.0/1024.0*4/time);
      }
      {
        MPI_Barrier(MPI_COMM_WORLD);
        double time = MPI_Wtime();
        if(myid==0)MPI_Send(senddata,N,MPI_INT,5,0,MPI_COMM_WORLD);
        if(myid==5)MPI_Recv(recvdata,N,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Barrier(MPI_COMM_WORLD);
        time = MPI_Wtime()-time;
        printf("within-node time %e data %e GB B/W %e (GB/s)\n",time,N/1024.0/1024.0/1024.0*4,N/1024.0/1024.0/1024.0*4/time);
      }
    }
    return;*/

  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("\nREDUCTION MAPPINGS\n");
  reducemap();
  MPI_Barrier(MPI_COMM_WORLD);
  time = MPI_Wtime();
  if(myid==0)printf("\nCONSTRUCT PROJECTION MATRIX\n");
  {
    int *rownz = new int[raynumout];
    #pragma omp parallel for
    for(int k = 0; k < raynumout; k++){
      double rho = raycoorout[k].real();
      double theta = raycoorout[k].imag();
      rownz[k] = 0;
      for(int tile = spatstart[myid_data]; tile < spatstart[myid_data]+numspats[myid_data]; tile++){
        double domain[4];
        domain[0]=spatll[tile].real();
        domain[1]=domain[0]+spatsize;
        domain[2]=spatll[tile].imag();
        domain[3]=domain[2]+spatsize;
        //REMOVE SPATIAL EDGE CONDITION
        if(domain[1] > xstart+numx)domain[1]=xstart+numx;
        if(domain[3] > ystart+numy)domain[3]=ystart+numy;
        findnumpix(theta,rho,&domain[0],&rownz[k]);
      }
    }
    long *rowdispl = new long[raynumout+1];
    rowdispl[0] = 0;
    for(int k = 1; k < raynumout+1; k++)
      rowdispl[k] = rowdispl[k-1]+rownz[k-1];
    delete[] rownz;
    long rownztot = rowdispl[raynumout];
    long rownzall = rownztot;
    MPI_Allreduce(MPI_IN_PLACE,&rownzall,1,MPI_LONG,MPI_SUM,MPI_COMM_DATA);
    long *rownztots = new long[numproc_data];
    MPI_Allgather(&rownztot,1,MPI_LONG,rownztots,1,MPI_LONG,MPI_COMM_DATA);
    long rownztotmin = rownztots[0];
    long rownztotmax = rownztots[0];
    for(int p = 0; p < numproc_data; p++){
      if(rownztotmin>rownztots[p])rownztotmin=rownztots[p];
      if(rownztotmax<rownztots[p])rownztotmax=rownztots[p];
    }
    if(myid==0){
      printf("CSR STORAGE: %ld (%f GB)\n",rownzall,rownzall*(sizeof(MATPREC)+sizeof(int))/1.0e9);
      for(int p = 0; p < numproc_data; p++)
        printf("proc: %d rownztot: %ld (%f GB)\n",p,rownztots[p],rownztots[p]/1.0e9*(sizeof(MATPREC)+sizeof(int)));
      printf("rownztotmin: %ld rownztotmax: %ld imbalance: %f\n",rownztotmin,rownztotmax,rownztotmax/((double)rownzall/numproc_data));
    }
    delete[] rownztots;
    int *rowindex = new int[rownztot];
    #pragma omp parallel for
    for(int k = 0; k < raynumout; k++){
      double rho = raycoorout[k].real();
      double theta = raycoorout[k].imag();
      long start = rowdispl[k];
      for(int tile = spatstart[myid_data]; tile < spatstart[myid_data]+numspats[myid_data]; tile++){
        double domain[4];
        domain[0]=spatll[tile].real();
        domain[1]=domain[0]+spatsize;
        domain[2]=spatll[tile].imag();
        domain[3]=domain[2]+spatsize;
        //REMOVE SPATIAL EDGE CONDITION
        if(domain[1] > xstart+numx)domain[1]=xstart+numx;
        if(domain[3] > ystart+numy)domain[3]=ystart+numy;
        int offset = (tile-spatstart[myid_data])*spatsize*spatsize;
        int pixtemp = 0;
        findpixind(theta,rho,&domain[0],&pixtemp,offset,&rowindex[start]);
        start=start+pixtemp;
      }
    }
    proj_rownztot = rownztot;
    proj_rownzall = rownzall;
    proj_rowdispl = rowdispl;
    proj_rowindex = rowindex;
  }
      /*float *tempval = new float[proj_rowdispl[raynumout]];
      #pragma omp parallel for
      for(int n = 0; n < proj_rowdispl[raynumout]; n++)
        tempval[n] = 1.0;
      int *tempdispl = new int[raynumout+1];
      #pragma omp parallel for
      for(int n = 0; n < raynumout+1; n++)
        tempdispl[n] = proj_rowdispl[n];
      FILE *matf = fopen("XCT_ADS3_Hilbert.bin","wb");
      fwrite(tempdispl,sizeof(int),raynumout+1,matf);
      fwrite(proj_rowindex,sizeof(int),proj_rowdispl[raynumout],matf);
      fwrite(tempval,sizeof(float),proj_rowdispl[raynumout],matf);
      fclose(matf);
      delete[] tempval;
      delete[] tempdispl;*/
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("RAY-TRACING TIME: %e\n",MPI_Wtime()-time);
  time = MPI_Wtime();
  if(myid==0)printf("CONSTRUCT BACKPROJECTION MATRIX\n");
  {
    long *csrRowInd = new long[proj_rownztot];
    int *inter = new int[(numthreads+1)*mynumpix];
    int *intra = new int[proj_rownztot];
    #pragma omp parallel for
    for(int k = 0; k < raynumout; k++)
      for(long n = proj_rowdispl[k]; n < proj_rowdispl[k+1]; n++)
        csrRowInd[n] = k;
    #pragma omp parallel for
    for(int n = 0; n < (numthreads+1)*mynumpix; n++)
      inter[n] = 0;
    #pragma omp parallel for
    for(long n = 0; n < proj_rownztot; n++){
      intra[n] = inter[(omp_get_thread_num()+1)*mynumpix+proj_rowindex[n]];
      inter[(omp_get_thread_num()+1)*mynumpix+proj_rowindex[n]]++;
    }
    #pragma omp parallel for
    for(int m = 0; m < mynumpix; m++)
      for(int t = 1; t < numthreads+1; t++)
        inter[t*mynumpix+m] = inter[t*mynumpix+m]+inter[(t-1)*mynumpix+m];
    long *rowdispl = new long[mynumpix+1];
    rowdispl[0] = 0;
    for(int m = 1; m < mynumpix+1; m++)
      rowdispl[m] = rowdispl[m-1] + inter[numthreads*mynumpix+m-1];
    long rownztot = rowdispl[mynumpix];
    int *rowindex = new int[rownztot];
    #pragma omp parallel for
    for(long n = 0; n < rownztot; n++){
      rowindex[rowdispl[proj_rowindex[n]]+inter[omp_get_thread_num()*mynumpix+proj_rowindex[n]]+intra[n]] = csrRowInd[n];
    }
    delete[] inter;
    delete[] intra;
    delete[] csrRowInd;
    back_rowdispl = rowdispl;
    back_rowindex = rowindex;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("TRANSPOSITION TIME: %e\n",MPI_Wtime()-time);
  time = MPI_Wtime();
  if(myid==0)printf("\nBLOCKING PROJECTION MATRIX\n");
  {
    int *rowindex = proj_rowindex;
    long *rowdispl = proj_rowdispl;
    int blocksize = proj_blocksize;
    int buffsize = proj_buffsize;
    int numblocks = raynumout/blocksize;
    if(raynumout%blocksize)numblocks++;
    int numblocksall = numblocks;
    MPI_Allreduce(MPI_IN_PLACE,&numblocksall,1,MPI_INT,MPI_SUM,MPI_COMM_DATA);
    if(myid==0)printf("TOTAL NUMBER OF BLOCKS: %d BUFFSIZE: %d (%f KB)\n",numblocksall,buffsize,buffsize*sizeof(VECPREC)/1.0e3);
    int *numbuff = new int[numblocks];
    #pragma omp parallel
    {
      int *numint = new int[mynumpix];
      #pragma omp for
      for(int block = 0; block < numblocks; block++){
        for(int n = 0; n < mynumpix; n++)
          numint[n] = 0;
        for(int m = block*blocksize; m < (block+1)*blocksize && m < raynumout; m++)
          for(long n = rowdispl[m]; n < rowdispl[m+1]; n++)
            numint[rowindex[n]]++;
        int count = 0;
        for(int n = 0; n < mynumpix; n++)
          if(numint[n])count++;
        numbuff[block] = count/buffsize;
        if(count%buffsize)numbuff[block]++;
      }
      delete[] numint;
    }
    int *buffdispl = new int[numblocks+1];
    buffdispl[0] = 0;
    for(int block = 1; block < numblocks+1; block++)
      buffdispl[block] = buffdispl[block-1] + numbuff[block-1];
    int numbufftot = buffdispl[numblocks];
    int numbuffmax = numbuff[0];
    int numbuffmin = numbuff[0];
    for(int block = 0; block < numblocks; block++){
      if(numbuff[block]>numbuffmax)numbuffmax = numbuff[block];
      if(numbuff[block]<numbuffmin)numbuffmin = numbuff[block];
    }
    long numbuffall = numbufftot;
    MPI_Allreduce(MPI_IN_PLACE,&numbuffall,1,MPI_LONG,MPI_SUM,MPI_COMM_DATA);
    if(myid==0)printf("TOTAL NUMBER OF BUFFERS: %ld (%f PER BLOCK) MIN %d MAX %d BUFF PER BLOCK\n",numbuffall,numbuffall/(double)numblocksall,numbuffmin,numbuffmax);
    int *mapnz = new int[numbufftot];
    for(int n = 0; n < numbufftot; n++)
      mapnz[n] = 0;
    #pragma omp parallel
    {
      int *numint = new int[mynumpix];
      #pragma omp for
      for(int block = 0; block < numblocks; block++){
        for(int n = 0; n < mynumpix; n++)
          numint[n] = 0;
        for(int m = block*blocksize; m < (block+1)*blocksize && m < raynumout; m++)
          for(long n = rowdispl[m]; n < rowdispl[m+1]; n++)
            numint[rowindex[n]]++;
        int count = 0;
        for(int n = 0; n < mynumpix; n++)
          if(numint[n])count++;
        for(int buff = buffdispl[block]; buff < buffdispl[block+1]; buff++){
          if(count/buffsize){
            mapnz[buff] = buffsize;
            count -= buffsize;
          }
          else
            mapnz[buff] = count;
        }
      }
      delete[] numint;
    }
    int *mapdispl = new int[numbufftot+1];
    mapdispl[0] = 0;
    for(int buff = 1; buff < numbufftot+1; buff++)
      mapdispl[buff] = mapdispl[buff-1] + mapnz[buff-1];
    //delete[] mapnz;
    int mapnztot = mapdispl[numbufftot];
    long mapnzall = mapnztot;
    MPI_Allreduce(MPI_IN_PLACE,&mapnzall,1,MPI_LONG,MPI_SUM,MPI_COMM_DATA);
    if(myid==0)printf("BUFF MAP: %ld (%f GB) DATA REUSE: %f\n",mapnzall,mapnzall/1.0e9*sizeof(int),proj_rownzall/(double)mapnzall);
    int numwarp = blocksize/WARPSIZE*numbufftot;
    long numwarpall = numwarp;
    MPI_Allreduce(MPI_IN_PLACE,&numwarpall,1,MPI_LONG,MPI_SUM,MPI_COMM_DATA);
    if(myid==0)printf("TOTAL NUMBER OF WARPS: %ld (WARPSIZE %d)\n",numwarpall,WARPSIZE);
    int *buffmap = new int[mapnztot];
    int *warpnz = new int[numwarp];
    #pragma omp for
    for(int n = 0; n < numwarp; n++)
      warpnz[n] = 0;
    #pragma omp parallel
    {
      int *numint = new int[mynumpix];
      int *indcount = new int[blocksize*numbuffmax];
      #pragma omp for
      for(int block = 0; block < numblocks; block++){
        for(int n = 0; n < mynumpix; n++)
          numint[n] = 0;
        for(int m = block*blocksize; m < (block+1)*blocksize && m < raynumout; m++)
          for(long n = rowdispl[m]; n < rowdispl[m+1]; n++)
            numint[rowindex[n]]++;
        int count = 0;
        for(int n = 0; n < mynumpix; n++)
          if(numint[n]){
            int buffloc = count/buffsize;
            int mapind = mapdispl[(buffdispl[block]+buffloc)]+count%buffsize;
            buffmap[mapind] = n;
            numint[n] = buffloc;
            count++;
          }
        for(int n = 0; n < blocksize*numbuff[block]; n++)
          indcount[n] = 0;
        for(int m = block*blocksize; m < (block+1)*blocksize && m < raynumout; m++)
          for(long n = rowdispl[m]; n < rowdispl[m+1]; n++)
            indcount[blocksize*numint[rowindex[n]]+m%blocksize]++;
        for(int buff = buffdispl[block]; buff < buffdispl[block+1]; buff++){
          int buffloc = buff-buffdispl[block];
          for(int warp = 0; warp < blocksize/WARPSIZE; warp++){
            int warpmax = 0;
            for(int n = warp*WARPSIZE; n < (warp+1)*WARPSIZE; n++){
              int test = indcount[blocksize*buffloc+n];
              if(test > warpmax)warpmax = test;
            }
            warpnz[blocksize/WARPSIZE*buff+warp] = warpmax;
          }
        }
      }
      delete[] numint;
      delete[] indcount;
    }
    int *warpdispl = new int[numwarp+1];
    warpdispl[0] = 0;
    for(int warp = 1; warp < numwarp+1; warp++)
      warpdispl[warp] = warpdispl[warp-1]+warpnz[warp-1];
    int warpnztot = warpdispl[numwarp];
    long warpnzall = warpnztot;
    MPI_Allreduce(MPI_IN_PLACE,&warpnzall,1,MPI_LONG,MPI_SUM,MPI_COMM_DATA);
    int *warpnztots = new int[numproc_data];
    MPI_Allgather(&warpnztot,1,MPI_INT,warpnztots,1,MPI_INT,MPI_COMM_DATA);
    int warpnztotmin = warpnztots[0];
    int warpnztotmax = warpnztots[0];
    for(int p = 0; p < numproc_data; p++){
      if(warpnztotmin>warpnztots[p])warpnztotmin=warpnztots[p];
      if(warpnztotmax<warpnztots[p])warpnztotmax=warpnztots[p];
    }
    if(myid==0){
      printf("WARP ELL NZ: %ld (%f GB) OVERHEAD: %f EFFICIENCY: %f\n",warpnzall*(long)WARPSIZE,warpnzall*(double)WARPSIZE*(sizeof(MATPREC)+sizeof(unsigned short))/1.0e9,warpnzall*(double)WARPSIZE/proj_rownzall,warpnzall*(double)WARPSIZE*0.75/proj_rownzall);
      for(int p = 0; p < numproc_data; p++)
        printf("proc %d: warpnztot: %ld (%f GB)\n",p,warpnztots[p]*(long)WARPSIZE,warpnztots[p]/1.0e9*WARPSIZE*(sizeof(MATPREC)+sizeof(unsigned short)));
      printf("warpnztotmin: %ld warpnztotmax: %ld imbalance: %f\n",warpnztotmin*(long)WARPSIZE,warpnztotmax*(long)WARPSIZE,warpnztotmax/((double)warpnzall/numproc_data));
    }
    delete[] warpnztots;
    #ifdef MATRIX
    matrix *warpindval = new matrix[warpnztot*(long)WARPSIZE];
    #else
    unsigned short *warpindex = new unsigned short[warpnztot*(long)WARPSIZE];
    #endif
    bool *warpindextag = new bool[warpnztot*(long)WARPSIZE];
    #pragma omp parallel for
    for(long n = 0; n < warpnztot*(long)WARPSIZE; n++){
      #ifdef MATRIX
      warpindval[n].ind = 0;
      #else
      warpindex[n] = 0;
      #endif
      warpindextag[n] = false;
    }
    #pragma omp parallel
    {
      int *numint = new int[mynumpix];
      int *numind = new int[mynumpix];
      int *indcount = new int[blocksize*numbuffmax];
      #pragma omp for
      for(int block = 0; block < numblocks; block++){
        for(int n = 0; n < mynumpix; n++)
          numint[n] = 0;
        for(int m = block*blocksize; m < (block+1)*blocksize && m < raynumout; m++)
          for(long n = rowdispl[m]; n < rowdispl[m+1]; n++)
            numint[rowindex[n]]++;
        int count = 0;
        for(int n = 0; n < mynumpix; n++)
          if(numint[n]){
            int buffloc = count/buffsize;
            numint[n] = buffloc;
            numind[n] = count%buffsize;
            count++;
          }
        for(int n = 0; n < blocksize*numbuff[block]; n++)
          indcount[n] = 0;
        for(int m = block*blocksize; m < (block+1)*blocksize && m < raynumout; m++)
          for(long n = rowdispl[m]; n < rowdispl[m+1]; n++){
            int buffloc = numint[rowindex[n]];
            int *count = indcount+blocksize*buffloc;
            int indloc = m%blocksize;
            int warp = ((buffdispl[block]+buffloc)*blocksize+indloc)/WARPSIZE;
            long ind = (warpdispl[warp]+count[indloc])*(long)WARPSIZE+m%WARPSIZE;
            #ifdef MATRIX
            warpindval[ind].ind = numind[rowindex[n]];
            #else
            warpindex[ind] = numind[rowindex[n]];
            #endif
            warpindextag[ind] = true;
            count[indloc]++;
          }
      }
      delete[] numint;
      delete[] numind;
      delete[] indcount;
    }
    delete[] numbuff;
    delete[] rowdispl;
    delete[] rowindex;
    proj_numblocks = numblocks;
    proj_numbufftot = numbufftot;
    proj_numbuffall = numbuffall;
    proj_buffdispl = buffdispl;
    proj_mapnztot = mapnztot;
    proj_mapnzall = mapnzall;
    proj_mapdispl = mapdispl;
    proj_mapnz = mapnz;
    proj_buffmap = buffmap;
    proj_warpnztot = warpnztot;
    proj_warpnzall = warpnzall;
    proj_warpdispl = warpdispl;
    #ifdef MATRIX
    proj_warpindval = warpindval;
    #else
    proj_warpindex = warpindex;
    #endif
    proj_warpindextag = warpindextag;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  time = MPI_Wtime();
  if(myid==0)printf("\nBLOCKING BACKPROJECTION MATRIX\n");
  {
    int *rowindex = back_rowindex;
    long *rowdispl = back_rowdispl;
    int blocksize = back_blocksize;
    int buffsize = back_buffsize;
    int numblocks = mynumpix/blocksize;
    if(mynumpix%blocksize)numblocks++;
    int numblocksall = numblocks;
    MPI_Allreduce(MPI_IN_PLACE,&numblocksall,1,MPI_INTEGER,MPI_SUM,MPI_COMM_DATA);
    if(myid==0)printf("TOTAL NUMBER OF BLOCKS: %d BUFFSIZE: %d (%f KB)\n",numblocksall,buffsize,buffsize*sizeof(VECPREC)/1.0e3);
    int *numbuff = new int[numblocks];
    #pragma omp parallel
    {
      int *numint = new int[raynumout];
      #pragma omp for
      for(int block = 0; block < numblocks; block++){
        for(int n = 0; n < raynumout; n++)
          numint[n] = 0;
        for(int m = block*blocksize; m < (block+1)*blocksize && m < mynumpix; m++)
          for(long n = rowdispl[m]; n < rowdispl[m+1]; n++)
            numint[rowindex[n]]++;
        int count = 0;
        for(int n = 0; n < raynumout; n++)
          if(numint[n])count++;
        numbuff[block] = count/buffsize;
        if(count%buffsize)numbuff[block]++;
      }
      delete[] numint;
    }
    int *buffdispl = new int[numblocks+1];
    buffdispl[0] = 0;
    for(int block = 1; block < numblocks+1; block++)
      buffdispl[block] = buffdispl[block-1] + numbuff[block-1];
    int numbufftot = buffdispl[numblocks];
    int numbuffmax = numbuff[0];
    int numbuffmin = numbuff[0];
    for(int block = 0; block < numblocks; block++){
      if(numbuff[block]>numbuffmax)numbuffmax = numbuff[block];
      if(numbuff[block]<numbuffmin)numbuffmin = numbuff[block];
    }
    long numbuffall = numbufftot;
    MPI_Allreduce(MPI_IN_PLACE,&numbuffall,1,MPI_LONG,MPI_SUM,MPI_COMM_DATA);
    if(myid==0)printf("TOTAL NUMBER OF BUFFERS: %ld (%f PER BLOCK) MIN %d MAX %d BUFF PER BLOCK\n",numbuffall,numbuffall/(double)numblocksall,numbuffmin,numbuffmax);
    int *mapnz = new int[numbufftot];
    for(int n = 0; n < numbufftot; n++)
      mapnz[n] = 0;
    #pragma omp parallel
    {
      int *numint = new int[raynumout];
      #pragma omp for
      for(int block = 0; block < numblocks; block++){
        for(int n = 0; n < raynumout; n++)
          numint[n] = 0;
        for(int m = block*blocksize; m < (block+1)*blocksize && m < mynumpix; m++)
          for(long n = rowdispl[m]; n < rowdispl[m+1]; n++)
            numint[rowindex[n]]++;
        int count = 0;
        for(int n = 0; n < raynumout; n++)
          if(numint[n])count++;
        for(int buff = buffdispl[block]; buff < buffdispl[block+1]; buff++){
          if(count/buffsize){
            mapnz[buff] = buffsize;
            count -= buffsize;
          }
          else
            mapnz[buff] = count;
        }
      }
      delete[] numint;
    }
    int *mapdispl = new int[numbufftot+1];
    mapdispl[0] = 0;
    for(int buff = 1; buff < numbufftot+1; buff++)
      mapdispl[buff] = mapdispl[buff-1] + mapnz[buff-1];
    //delete[] mapnz;
    int mapnztot = mapdispl[numbufftot];
    long mapnzall = mapnztot;
    MPI_Allreduce(MPI_IN_PLACE,&mapnzall,1,MPI_LONG,MPI_SUM,MPI_COMM_DATA);
    if(myid==0)printf("BUFF MAP: %ld (%f GB) DATA REUSE: %f\n",mapnzall,mapnzall/1.0e9*sizeof(int),proj_rownzall/(double)mapnzall);
    int numwarp = blocksize/WARPSIZE*numbufftot;
    long numwarpall = numwarp;
    MPI_Allreduce(MPI_IN_PLACE,&numwarpall,1,MPI_LONG,MPI_SUM,MPI_COMM_DATA);
    if(myid==0)printf("TOTAL NUMBER OF WARPS: %ld (WARPSIZE %d)\n",numwarpall,WARPSIZE);
    int *buffmap = new int[mapnztot];
    int *warpnz = new int[numwarp];
    #pragma omp for
    for(int n = 0; n < numwarp; n++)
      warpnz[n] = 0;
    #pragma omp parallel
    {
      int *numint = new int[raynumout];
      int *indcount = new int[blocksize*numbuffmax];
      #pragma omp for
      for(int block = 0; block < numblocks; block++){
        for(int n = 0; n < raynumout; n++)
          numint[n] = 0;
        for(int m = block*blocksize; m < (block+1)*blocksize && m < mynumpix; m++)
          for(long n = rowdispl[m]; n < rowdispl[m+1]; n++)
            numint[rowindex[n]]++;
        int count = 0;
        for(int n = 0; n < raynumout; n++)
          if(numint[n]){
            int buffloc = count/buffsize;
            int mapind = mapdispl[(buffdispl[block]+buffloc)]+count%buffsize;
            buffmap[mapind] = n;
            numint[n] = buffloc;
            count++;
          }
        for(int n = 0; n < blocksize*numbuff[block]; n++)
          indcount[n] = 0;
        for(int m = block*blocksize; m < (block+1)*blocksize && m < mynumpix; m++)
          for(long n = rowdispl[m]; n < rowdispl[m+1]; n++)
            indcount[blocksize*numint[rowindex[n]]+m%blocksize]++;
        for(int buff = buffdispl[block]; buff < buffdispl[block+1]; buff++){
          int buffloc = buff-buffdispl[block];
          for(int warp = 0; warp < blocksize/WARPSIZE; warp++){
            int warpmax = 0;
            for(int n = warp*WARPSIZE; n < (warp+1)*WARPSIZE; n++){
              int test = indcount[blocksize*buffloc+n];
              if(test > warpmax)warpmax = test;
            }
            warpnz[blocksize/WARPSIZE*buff+warp] = warpmax;
          }
        }
      }
      delete[] numint;
      delete[] indcount;
    }
    int *warpdispl = new int[numwarp+1];
    warpdispl[0] = 0;
    for(int warp = 1; warp < numwarp+1; warp++)
      warpdispl[warp] = warpdispl[warp-1]+warpnz[warp-1];
    int warpnztot = warpdispl[numwarp];
    long warpnzall = warpnztot;
    MPI_Allreduce(MPI_IN_PLACE,&warpnzall,1,MPI_LONG,MPI_SUM,MPI_COMM_DATA);
    int *warpnztots = new int[numproc_data];
    MPI_Allgather(&warpnztot,1,MPI_INT,warpnztots,1,MPI_INT,MPI_COMM_DATA);
    int warpnztotmin = warpnztots[0];
    int warpnztotmax = warpnztots[0];
    for(int p = 0; p < numproc_data; p++){
      if(warpnztotmin>warpnztots[p])warpnztotmin=warpnztots[p];
      if(warpnztotmax<warpnztots[p])warpnztotmax=warpnztots[p];
    }
    if(myid==0){
      printf("WARP ELL NZ: %ld (%f GB) OVERHEAD: %f EFFICIENCY: %f\n",warpnzall*(long)WARPSIZE,warpnzall*(double)WARPSIZE*(sizeof(MATPREC)+sizeof(unsigned short))/1.0e9,warpnzall*(double)WARPSIZE/proj_rownzall,warpnzall*(double)WARPSIZE*0.75/proj_rownzall);
      for(int p = 0; p < numproc_data; p++)
        printf("proc %d: warpnztot: %ld (%f GB)\n",p,warpnztots[p]*(long)WARPSIZE,warpnztots[p]/1.0e9*WARPSIZE*(sizeof(MATPREC)+sizeof(unsigned short)));
      printf("warpnztotmin: %ld warpnztotmax: %ld imbalance: %f\n",warpnztotmin*(long)WARPSIZE,warpnztotmax*(long)WARPSIZE,warpnztotmax/((double)warpnzall/numproc_data));
    }
    delete[] warpnztots;
    #ifdef MATRIX
    matrix *warpindval = new matrix[warpnztot*(long)WARPSIZE];
    #else
    unsigned short *warpindex = new unsigned short[warpnztot*(long)WARPSIZE];
    #endif
    bool *warpindextag = new bool[warpnztot*(long)WARPSIZE];
    #pragma omp parallel for
    for(long n = 0; n < warpnztot*(long)WARPSIZE; n++){
      #ifdef MATRIX
      warpindval[n].ind = 0;
      #else
      warpindex[n] = 0;
      #endif
      warpindextag[n] = false;
    }
    #pragma omp parallel
    {
      int *numint = new int[raynumout];
      int *numind = new int[raynumout];
      int *indcount = new int[blocksize*numbuffmax];
      #pragma omp for
      for(int block = 0; block < numblocks; block++){
        for(int n = 0; n < raynumout; n++)
          numint[n] = 0;
        for(int m = block*blocksize; m < (block+1)*blocksize && m < mynumpix; m++)
          for(long n = rowdispl[m]; n < rowdispl[m+1]; n++)
            numint[rowindex[n]]++;
        int count = 0;
        for(int n = 0; n < raynumout; n++)
          if(numint[n]){
            int buffloc = count/buffsize;
            numint[n] = buffloc;
            numind[n] = count%buffsize;
            count++;
          }
        for(int n = 0; n < blocksize*numbuff[block]; n++)
          indcount[n] = 0;
        for(int m = block*blocksize; m < (block+1)*blocksize && m < mynumpix; m++)
          for(long n = rowdispl[m]; n < rowdispl[m+1]; n++){
            int buffloc = numint[rowindex[n]];
            int *count = indcount+blocksize*buffloc;
            int indloc = m%blocksize;
            int warp = ((buffdispl[block]+buffloc)*blocksize+indloc)/WARPSIZE;
            long ind = (warpdispl[warp]+count[indloc])*(long)WARPSIZE+m%WARPSIZE;
            #ifdef MATRIX
            warpindval[ind].ind = numind[rowindex[n]];
            #else
            warpindex[ind] = numind[rowindex[n]];
            #endif
            warpindextag[ind] = true;
            count[indloc]++;
          }
      }
      delete[] numint;
      delete[] numind;
      delete[] indcount;
    }
    delete[] numbuff;
    delete[] rowdispl;
    delete[] rowindex;
    back_numblocks = numblocks;
    back_numbufftot = numbufftot;
    back_numbuffall = numbuffall;
    back_buffdispl = buffdispl;
    back_mapnztot = mapnztot;
    back_mapnzall = mapnzall;
    back_mapdispl = mapdispl;
    back_mapnz = mapnz;
    back_buffmap = buffmap;
    back_warpnztot = warpnztot;
    back_warpnzall = warpnzall;
    back_warpdispl = warpdispl;
    #ifdef MATRIX
    back_warpindval = warpindval;
    #else
    back_warpindex = warpindex;
    #endif
    back_warpindextag = warpindextag;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("BLOCKING TIME: %e\n",MPI_Wtime()-time);
  time = MPI_Wtime();
  if(myid==0)printf("\nFILL PROJECTION MATRIX\n");
  {
    #ifndef MATRIX
    proj_warpvalue = new MATPREC[proj_warpnztot*(long)WARPSIZE];
    #endif
    #pragma omp parallel for
    for(long n = 0; n < proj_warpnztot*(long)WARPSIZE; n++)
      #ifdef MATRIX
      proj_warpindval[n].val = 0.0;
      #else
      proj_warpvalue[n] = 0.0;
      #endif
    int underflow = 0;
    #pragma omp parallel for
    for(int block = 0; block < proj_numblocks; block++){
      for(int ray = block*proj_blocksize; ray < (block+1)*proj_blocksize && ray < raynumout; ray++){
        double rho = raycoorout[ray].real();
        double theta = raycoorout[ray].imag();
        int n = ray%proj_blocksize;
        for(int buff = proj_buffdispl[block]; buff < proj_buffdispl[block+1]; buff++){
          int warp = (buff*proj_blocksize+n)/WARPSIZE;
          for(int row = proj_warpdispl[warp]; row < proj_warpdispl[warp+1]; row++){
            long ind = row*(long)WARPSIZE+n%WARPSIZE;
            if(proj_warpindextag[ind]){
              #ifdef MATRIX
              int mapind = proj_mapdispl[buff]+proj_warpindval[ind].ind;
              #else
              int mapind = proj_mapdispl[buff]+proj_warpindex[ind];
              #endif
              int pixind = proj_buffmap[mapind];
              double domain[4];
              domain[0]=pixcoor[pixind].real()-0.5;
              domain[1]=domain[0]+1.0;
              domain[2]=pixcoor[pixind].imag()-0.5;
              domain[3]=domain[2]+1.0;
              double temp = 0;
              findlength(theta,rho,domain,&temp);
              temp *= pixsize;
              if((MATPREC)temp==0.0)
                #pragma omp atomic
                underflow++;
              #ifdef MATRIX
              proj_warpindval[ind].val = temp;
              #else
              proj_warpvalue[ind] = temp;
              #endif
            }
          }
        }
      }
    }
    double rowmax = numx*pixsize*sqrt(2);
    if(myid==0)printf("rowmax: %e underflow: %d\n",rowmax,underflow);
    proj_rowmax = rowmax;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("TIME: %e\n",MPI_Wtime()-time);
  time = MPI_Wtime();
  if(myid==0)printf("FILL BACKPROJECTION MATRIX\n");
  {
    #ifndef MATRIX
    back_warpvalue = new MATPREC[back_warpnztot*(long)WARPSIZE];
    #endif
    #pragma omp parallel for
    for(long n = 0; n < back_warpnztot*(long)WARPSIZE; n++)
      #ifdef MATRIX
      back_warpindval[n].val = 0.0;
      #else
      back_warpvalue[n] = 0.0;
      #endif
    double rowmax = 0.0;
    int underflow = 0;
    #pragma omp parallel for
    for(int block = 0; block < back_numblocks; block++){
      for(int pix = block*back_blocksize; pix < (block+1)*back_blocksize && pix < mynumpix; pix++){
        double reduce = 0.0;
        double domain[4];
        domain[0]=pixcoor[pix].real()-0.5;
        domain[1]=domain[0]+1.0;
        domain[2]=pixcoor[pix].imag()-0.5;
        domain[3]=domain[2]+1.0;
        int n = pix%back_blocksize;
        for(int buff = back_buffdispl[block]; buff < back_buffdispl[block+1]; buff++){
          int warp = (buff*back_blocksize+n)/WARPSIZE;
          for(int row = back_warpdispl[warp]; row < back_warpdispl[warp+1]; row++){
            long ind = row*(long)WARPSIZE+n%WARPSIZE;
            if(back_warpindextag[ind]){
              #ifdef MATRIX
              int mapind = back_mapdispl[buff]+back_warpindval[ind].ind;
              #else
              int mapind = back_mapdispl[buff]+back_warpindex[ind];
              #endif
              int rayind = back_buffmap[mapind];
              double rho = raycoorout[rayind].real();
              double theta = raycoorout[rayind].imag();
              double temp = 0;
              findlength(theta,rho,domain,&temp);
              temp *= pixsize;
              if((MATPREC)temp==0.0)
                #pragma omp atomic
                underflow++;
              #ifdef MATRIX
              back_warpindval[ind].val = temp;
              #else
              back_warpvalue[ind] = temp;
              #endif
              reduce += temp;
            }
          }
        }
        if(reduce>rowmax)rowmax=reduce;
      }
    }
    MPI_Allreduce(MPI_IN_PLACE,&rowmax,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_DATA);
    if(myid==0)printf("rowmax: %e underflow: %d\n",rowmax,underflow);
    back_rowmax = rowmax;
  }
  delete[] raycoorout;
  delete[] raycoorinc;
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("TIME: %e\n",MPI_Wtime()-time);
}

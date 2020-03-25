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

extern int *raysendstart;
extern int *rayrecvstart;
extern int *raysendcount;
extern int *rayrecvcount;

extern int *rayraystart;
extern int *rayrayind;

extern int *rayrecvlist;

extern int proj_blocksize;
extern int proj_buffsize;
extern int back_blocksize;
extern int back_buffsize;



double proj_rowmax;
int proj_rownztot;
long proj_rownzall;
int *proj_rowdispl;
int *proj_rowindex;
int proj_numblocks;
int proj_numbufftot;
int proj_numbuffall;
int *proj_buffdispl;
int proj_mapnztot;
int proj_mapnzall;
int *proj_mapdispl;
int *proj_mapnz;
int *proj_buffmap;
int proj_warpnztot;
long proj_warpnzall;
int *proj_warpdispl;
unsigned short *proj_warpindex;
bool *proj_warpindextag;
MATPREC *proj_warpvalue;

double back_rowmax;
int *back_rowdispl;
int *back_rowindex;
int back_numblocks;
int back_numbufftot;
int back_numbuffall;
int *back_buffdispl;
int back_mapnztot;
int back_mapnzall;
int *back_mapdispl;
int *back_mapnz;
int *back_buffmap;
int back_warpnztot;
long back_warpnzall;
int *back_warpdispl;
unsigned short *back_warpindex;
bool *back_warpindextag;
MATPREC *back_warpvalue;

int *rayglobalind;
int *pixglobalind;
int *raymesind;
int *pixobjind;
int *objglobalind;

complex<double> *pixcoor;
complex<double> *raycoor;

int *numpixs;
int *numrays;
int *pixstart;
int *raystart;

void preproc(){

  int numproc;
  int myid;
  MPI_Comm_size(MPI_COMM_WORLD,&numproc);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  int numthreads;
  #pragma omp parallel
  if(omp_get_thread_num()==0)numthreads=omp_get_num_threads();

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
    double x = xstart+xtile*spatsize*pixsize;
    double y = ystart+ytile*spatsize*pixsize;
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
  int numspats[numproc];
  int numspecs[numproc];
  int spatstart[numproc];
  int specstart[numproc];
  numpixs = new int[numproc];
  numrays = new int [numproc];
  pixstart = new int[numproc];
  raystart = new int[numproc];
  int myspattemp = (numspattile/numproc)*numproc;
  int myspectemp = (numspectile/numproc)*numproc;
  for(int p = 0; p < numproc; p++){
    numspats[p] = numspattile/numproc;
    numspecs[p] = numspectile/numproc;
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
  for(int p = 1; p < numproc; p++){
    spatstart[p] = spatstart[p-1] + numspats[p-1];
    specstart[p] = specstart[p-1] + numspecs[p-1];
  }
  for(int p = 0; p < numproc; p++){
    numpixs[p] = numspats[p]*spatsize*spatsize;
    numrays[p] = numspecs[p]*specsize*specsize;
  }
  pixstart[0] = 0;
  raystart[0] = 0;
  for(int p = 1; p < numproc; p++){
    pixstart[p] = pixstart[p-1] + numpixs[p-1];
    raystart[p] = raystart[p-1] + numrays[p-1];
  }
  mynumpix = numpixs[myid];
  mynumray = numrays[myid];
  int maxnumpix = numpixs[0];
  int maxnumray = numrays[0];
  int minnumpix = numpixs[0];
  int minnumray = numrays[0];
  for(int p = 0; p < numproc; p++){
    if(numpixs[p]>maxnumpix)maxnumpix=numpixs[p];
    if(numrays[p]>maxnumray)maxnumray=numrays[p];
    if(numpixs[p]<minnumpix)minnumpix=numpixs[p];
    if(numrays[p]<minnumray)minnumray=numrays[p];
  }
  if(myid==0){
    for(int p = 0; p < numproc; p++)
      printf("proc: %d numspats: %d numpixs: %d (%d blocks) /  numspecs: %d numrays: %d (%d blocks)\n",p,numspats[p],numpixs[p],numpixs[p]/back_blocksize,numspecs[p],numrays[p],numrays[p]/proj_blocksize);
    printf("minnumpix: %d maxnumpix: %d imbalance: %f\n",minnumpix,maxnumpix,maxnumpix/((double)(numpix)/numproc));
    printf("minnumray: %d maxnumray: %d imbalance: %f\n",minnumray,maxnumray,maxnumray/((double)(numray)/numproc));
  }
  if(myid==0)printf("FILL PIXELS AND RAYS\n");
  //PLACE PIXELS
  pixcoor = new complex<double>[mynumpix];
  pixglobalind = new int[mynumpix];
  pixobjind = new int[mynumpix];
  #pragma omp parallel for
  for(int pix = 0; pix < mynumpix; pix++){
    int tile = pix/(spatsize*spatsize);
    int pixloc = pix%(spatsize*spatsize);
    int pixlocy = pixloc/spatsize;
    int pixlocx = pixloc%spatsize;
    int  ind = tile*spatsize*spatsize + xy2d(spatsize,pixlocx,pixlocy);
    double x = spatll[spatstart[myid]+tile].real()+pixsize/2+pixlocx*pixsize;
    double y = spatll[spatstart[myid]+tile].imag()+pixsize/2+pixlocy*pixsize;
    pixcoor[ind] = complex<double>(x,y);
    //GLOBAL SPATIAL INDEX (EXTENDED)
    int xglobalind = (int)((x-xstart)/pixsize);
    int yglobalind = (int)((y-ystart)/pixsize);
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
  fclose(thetaf);
  //PLACE RAYS
  raycoor = new complex<double>[mynumray];
  rayglobalind = new int[mynumray];
  raymesind = new int[mynumray];
  #pragma omp parallel for
  for(int ray = 0; ray < mynumray; ray++){
    int tile = ray/(specsize*specsize);
    int rayloc = ray%(specsize*specsize);
    int raylocthe = rayloc/specsize;
    int raylocrho = rayloc%specsize;
    int ind = tile*specsize*specsize + xy2d(specsize,raylocrho,raylocthe);
    double rho = specll[specstart[myid]+tile].real()+0.5+raylocrho;
    double the = specll[specstart[myid]+tile].imag()+raylocthe*M_PI/numt;
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
  //delete[] mestheta;
  delete[] specll;
  if(myid==0)printf("DOMAIN PARTITIONING\n");
  rayrecvcount = new int[numproc];
  raysendcount = new int[numproc];
  rayrecvstart = new int[numproc];
  raysendstart = new int[numproc];
  double *lengthtemp = new double[mynumray];
  int **rayrecvtemp = new int*[numproc];
  for(int p = 0; p < numproc; p++){
    #pragma omp parallel for
    for(int k = 0; k < mynumray; k++){
      lengthtemp[k] = 0;
      double rho = raycoor[k].real();
      double theta = raycoor[k].imag();
      for(int tile = spatstart[p]; tile < spatstart[p]+numspats[p]; tile++){
        double domain[4];
        domain[0]=spatll[tile].real();
        domain[1]=domain[0]+spatsize*pixsize;
        domain[2]=spatll[tile].imag();
        domain[3]=domain[2]+spatsize*pixsize;
        //REMOVE SPATIAL EDGE CONDITION
        if(domain[1] > xstart+numx*pixsize)domain[1]=xstart+numx*pixsize;
        if(domain[3] > ystart+numy*pixsize)domain[3]=ystart+numy*pixsize;
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
  MPI_Alltoall(rayrecvcount,1,MPI_INTEGER,raysendcount,1,MPI_INTEGER,MPI_COMM_WORLD);
  rayrecvstart[0] = 0;
  raysendstart[0] = 0;
  for(int p = 1; p < numproc; p++){
    rayrecvstart[p] = rayrecvstart[p-1] + rayrecvcount[p-1];
    raysendstart[p] = raysendstart[p-1] + raysendcount[p-1];
  }
  raynuminc = rayrecvstart[numproc-1]+rayrecvcount[numproc-1];
  raynumout = raysendstart[numproc-1]+raysendcount[numproc-1];
  long raynumincall = raynuminc;
  raynumoutall = raynumout;
  MPI_Allreduce(MPI_IN_PLACE,&raynumincall,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&raynumoutall,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
  int *raynumouts = new int[numproc];
  int *raynumincs = new int[numproc];
  MPI_Allgather(&raynumout,1,MPI_INT,raynumouts,1,MPI_INT,MPI_COMM_WORLD);
  MPI_Allgather(&raynuminc,1,MPI_INT,raynumincs,1,MPI_INT,MPI_COMM_WORLD);
  int raynumoutmin = raynumouts[0];
  int raynumoutmax = raynumouts[0];
  int raynumincmin = raynumincs[0];
  int raynumincmax = raynumincs[0];
  for(int p = 0; p < numproc; p++){
    if(raynumoutmin>raynumouts[p])raynumoutmin=raynumouts[p];
    if(raynumoutmax<raynumouts[p])raynumoutmax=raynumouts[p];
    if(raynumincmin>raynumincs[p])raynumincmin=raynumincs[p];
    if(raynumincmax<raynumincs[p])raynumincmax=raynumincs[p];
  }
  if(myid==0){
    printf("total outgoing rays: %ld %fx (%f blocks av. %f per proc)\n",raynumoutall,raynumoutall/(double)(numt*numr),raynumoutall/(double)proj_blocksize,raynumoutall/(double)proj_blocksize/numproc);
    for(int p = 0; p < numproc; p++)
      printf("proc: %d raynumout: %d (%d blocks) raynuminc: %d (%d blocks)\n",p,raynumouts[p],raynumouts[p]/proj_blocksize,raynumincs[p],raynumincs[p]/back_blocksize);
    printf("raynumoutmin: %d raynumoutmax: %d imbalance: %f\n",raynumoutmin,raynumoutmax,raynumoutmax/((double)raynumoutall/numproc));
    printf("raynumincmin: %d raynumincmax: %d imbalance: %f\n",raynumincmin,raynumincmax,raynumincmax/((double)raynumincall/numproc));
  }
  delete[] raynumouts;
  delete[] raynumincs;
  int *raysendlist = new int[raynumout];
  rayrecvlist = new int[raynuminc];
  for(int p = 0; p < numproc; p++){
    #pragma omp parallel for
    for(int k = 0; k < rayrecvcount[p]; k++)
      rayrecvlist[rayrecvstart[p]+k] = rayrecvtemp[p][k];
    delete[] rayrecvtemp[p];
  }
  delete[] rayrecvtemp;
  MPI_Alltoallv(rayrecvlist,rayrecvcount,rayrecvstart,MPI_INTEGER,raysendlist,raysendcount,raysendstart,MPI_INTEGER,MPI_COMM_WORLD);
  //EXCHANGE RAY COORDINATES
  complex<double> *raycoorinc = new complex<double>[raynuminc];
  complex<double> *raycoorout = new complex<double>[raynumout];
  #pragma omp parallel for
  for(int k = 0; k < raynuminc; k++)
    raycoorinc[k] = raycoor[rayrecvlist[k]];
  MPI_Alltoallv(raycoorinc,rayrecvcount,rayrecvstart,MPI_DOUBLE_COMPLEX,raycoorout,raysendcount,raysendstart,MPI_DOUBLE_COMPLEX,MPI_COMM_WORLD);
  delete[] raycoor;
  //OBJECT MAPPING
  {
    extern int batchsize;
    int *pixobjinds;
    if(myid == 0){
      pixobjinds = new int[numpix];
      objglobalind = new int[numx*numy*batchsize];
    }
    MPI_Gatherv(pixobjind,mynumpix,MPI_INT,pixobjinds,numpixs,pixstart,MPI_INT,0,MPI_COMM_WORLD);
    if(myid==0){
      for(int p = 0; p < numproc; p++)
        #pragma omp parallel for
        for(int n = 0; n < numpixs[p]; n++){
          int ind = pixobjinds[pixstart[p]+n];
          if(ind > -1)
            for(int slice = 0; slice < batchsize; slice++)
              objglobalind[slice*numx*numy+ind] = batchsize*pixstart[p]+slice*numpixs[p]+n;
        }
      delete[] pixobjinds;
    }
  }
    /*{
      int N = 1000000;
      int *senddata = new int[N*numproc];
      int *recvdata = new int[N*numproc];
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
      for(int tile = spatstart[myid]; tile < spatstart[myid]+numspats[myid]; tile++){
        double domain[4];
        domain[0]=spatll[tile].real();
        domain[1]=domain[0]+spatsize*pixsize;
        domain[2]=spatll[tile].imag();
        domain[3]=domain[2]+spatsize*pixsize;
        //REMOVE SPATIAL EDGE CONDITION
        if(domain[1] > xstart+numx*pixsize)domain[1]=xstart+numx*pixsize;
        if(domain[3] > ystart+numy*pixsize)domain[3]=ystart+numy*pixsize;
        findnumpix(theta,rho,&domain[0],&rownz[k]);
      }
    }
    int *rowdispl = new int[raynumout+1];
    rowdispl[0] = 0;
    for(int k = 1; k < raynumout+1; k++)
      rowdispl[k] = rowdispl[k-1]+rownz[k-1];
    delete[] rownz;
    int rownztot = rowdispl[raynumout];
    long rownzall = rownztot;
    MPI_Allreduce(MPI_IN_PLACE,&rownzall,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    int *rownztots = new int[numproc];
    MPI_Allgather(&rownztot,1,MPI_INT,rownztots,1,MPI_INT,MPI_COMM_WORLD);
    int rownztotmin = rownztots[0];
    int rownztotmax = rownztots[0];
    for(int p = 0; p < numproc; p++){
      if(rownztotmin>rownztots[p])rownztotmin=rownztots[p];
      if(rownztotmax<rownztots[p])rownztotmax=rownztots[p];
    }
    if(myid==0){
      printf("CSR STORAGE: %ld (%f GB)\n",rownzall,rownzall*(sizeof(MATPREC)+sizeof(int))/1.0e9);
      for(int p = 0; p < numproc; p++)
        printf("proc: %d rownztot: %d (%f GB)\n",p,rownztots[p],rownztots[p]/1.0e9*(sizeof(MATPREC)+sizeof(int)));
        printf("rownztotmin: %d rownztotmax: %d imbalance: %f\n",rownztotmin,rownztotmax,rownztotmax/((double)rownzall/numproc));
    }
    delete[] rownztots;
    int *rowindex = new int[rownztot];
    #pragma omp parallel for
    for(int k = 0; k < raynumout; k++){
      double rho = raycoorout[k].real();
      double theta = raycoorout[k].imag();
      int start = rowdispl[k];
      for(int tile = spatstart[myid]; tile < spatstart[myid]+numspats[myid]; tile++){
        double domain[4];
        domain[0]=spatll[tile].real();
        domain[1]=domain[0]+spatsize*pixsize;
        domain[2]=spatll[tile].imag();
        domain[3]=domain[2]+spatsize*pixsize;
        //REMOVE SPATIAL EDGE CONDITION
        if(domain[1] > xstart+numx*pixsize)domain[1]=xstart+numx*pixsize;
        if(domain[3] > ystart+numy*pixsize)domain[3]=ystart+numy*pixsize;
        int offset = (tile-spatstart[myid])*spatsize*spatsize;
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
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("RAY-TRACING TIME: %e\n",MPI_Wtime()-time);
  time = MPI_Wtime();
  if(myid==0)printf("CONSTRUCT BACKPROJECTION MATRIX\n");
  {
    int *csrRowInd = new int[proj_rownztot];
    int *inter = new int[(numthreads+1)*mynumpix];
    int *intra = new int[proj_rownztot];
    #pragma omp parallel for
    for(int k = 0; k < raynumout; k++)
      for(int n = proj_rowdispl[k]; n < proj_rowdispl[k+1]; n++)
        csrRowInd[n] = k;
    #pragma omp parallel for
    for(int n = 0; n < (numthreads+1)*mynumpix; n++)
      inter[n] = 0;
    #pragma omp parallel for
    for(int n = 0; n < proj_rownztot; n++){
      intra[n] = inter[(omp_get_thread_num()+1)*mynumpix+proj_rowindex[n]];
      inter[(omp_get_thread_num()+1)*mynumpix+proj_rowindex[n]]++;
    }
    #pragma omp parallel for
    for(int m = 0; m < mynumpix; m++)
      for(int t = 1; t < numthreads+1; t++)
        inter[t*mynumpix+m] = inter[t*mynumpix+m]+inter[(t-1)*mynumpix+m];
    int *rowdispl = new int[mynumpix+1];
    rowdispl[0] = 0;
    for(int m = 1; m < mynumpix+1; m++)
      rowdispl[m] = rowdispl[m-1] + inter[numthreads*mynumpix+m-1];
    int rownztot = rowdispl[mynumpix];
    int *rowindex = new int[rownztot];
    #pragma omp parallel for
    for(int n = 0; n < rownztot; n++){
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
    int *rowdispl = proj_rowdispl;
    int blocksize = proj_blocksize;
    int buffsize = proj_buffsize;
    int numblocks = raynumout/blocksize;
    if(raynumout%blocksize)numblocks++;
    int numblocksall = numblocks;
    MPI_Allreduce(MPI_IN_PLACE,&numblocksall,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
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
          for(int n = rowdispl[m]; n < rowdispl[m+1]; n++)
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
    MPI_Allreduce(MPI_IN_PLACE,&numbuffall,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
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
          for(int n = rowdispl[m]; n < rowdispl[m+1]; n++)
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
    MPI_Allreduce(MPI_IN_PLACE,&mapnzall,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    if(myid==0)printf("BUFF MAP: %ld (%f GB) DATA REUSE: %f\n",mapnzall,mapnzall/1.0e9*sizeof(int),proj_rownzall/(double)mapnzall);
    int numwarp = blocksize/WARPSIZE*numbufftot;
    long numwarpall = numwarp;
    MPI_Allreduce(MPI_IN_PLACE,&numwarpall,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
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
          for(int n = rowdispl[m]; n < rowdispl[m+1]; n++)
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
          for(int n = rowdispl[m]; n < rowdispl[m+1]; n++)
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
    MPI_Allreduce(MPI_IN_PLACE,&warpnzall,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    int *warpnztots = new int[numproc];
    MPI_Allgather(&warpnztot,1,MPI_INT,warpnztots,1,MPI_INT,MPI_COMM_WORLD);
    int warpnztotmin = warpnztots[0];
    int warpnztotmax = warpnztots[0];
    for(int p = 0; p < numproc; p++){
      if(warpnztotmin>warpnztots[p])warpnztotmin=warpnztots[p];
      if(warpnztotmax<warpnztots[p])warpnztotmax=warpnztots[p];
    }
    if(myid==0){
      printf("WARP ELL NZ: %ld (%f GB) OVERHEAD: %f EFFICIENCY: %f\n",warpnzall*(long)WARPSIZE,warpnzall*(double)WARPSIZE*(sizeof(MATPREC)+sizeof(unsigned short))/1.0e9,warpnzall*(double)WARPSIZE/proj_rownzall,warpnzall*(double)WARPSIZE*0.75/proj_rownzall);
      for(int p = 0; p < numproc; p++)
        printf("proc %d: warpnztot: %d (%f GB)\n",p,warpnztots[p]*WARPSIZE,warpnztots[p]/1.0e9*WARPSIZE*(sizeof(MATPREC)+sizeof(unsigned short)));
      printf("warpnztotmin: %d warpnztotmax: %d imbalance: %f\n",warpnztotmin*WARPSIZE,warpnztotmax*WARPSIZE,warpnztotmax/((double)warpnzall/numproc));
    }
    delete[] warpnztots;
    unsigned short *warpindex = new unsigned short[warpnztot*WARPSIZE];
    bool *warpindextag = new bool[warpnztot*WARPSIZE];
    #pragma omp parallel for
    for(int n = 0; n < warpnztot*WARPSIZE; n++){
      warpindex[n] = 0;
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
          for(int n = rowdispl[m]; n < rowdispl[m+1]; n++)
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
          for(int n = rowdispl[m]; n < rowdispl[m+1]; n++){
            int buffloc = numint[rowindex[n]];
            int *count = indcount+blocksize*buffloc;
            int indloc = m%blocksize;
            int warp = ((buffdispl[block]+buffloc)*blocksize+indloc)/WARPSIZE;
            int ind = (warpdispl[warp]+count[indloc])*WARPSIZE+m%WARPSIZE;
            warpindex[ind] = numind[rowindex[n]];
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
    proj_warpindex = warpindex;
    proj_warpindextag = warpindextag;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  time = MPI_Wtime();
  if(myid==0)printf("\nBLOCKING BACKPROJECTION MATRIX\n");
  {
    int *rowindex = back_rowindex;
    int *rowdispl = back_rowdispl;
    int blocksize = back_blocksize;
    int buffsize = back_buffsize;
    int numblocks = mynumpix/blocksize;
    if(mynumpix%blocksize)numblocks++;
    int numblocksall = numblocks;
    MPI_Allreduce(MPI_IN_PLACE,&numblocksall,1,MPI_INTEGER,MPI_SUM,MPI_COMM_WORLD);
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
          for(int n = rowdispl[m]; n < rowdispl[m+1]; n++)
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
    MPI_Allreduce(MPI_IN_PLACE,&numbuffall,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
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
          for(int n = rowdispl[m]; n < rowdispl[m+1]; n++)
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
    MPI_Allreduce(MPI_IN_PLACE,&mapnzall,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    if(myid==0)printf("BUFF MAP: %ld (%f GB) DATA REUSE: %f\n",mapnzall,mapnzall/1.0e9*sizeof(int),proj_rownzall/(double)mapnzall);
    int numwarp = blocksize/WARPSIZE*numbufftot;
    long numwarpall = numwarp;
    MPI_Allreduce(MPI_IN_PLACE,&numwarpall,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
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
          for(int n = rowdispl[m]; n < rowdispl[m+1]; n++)
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
          for(int n = rowdispl[m]; n < rowdispl[m+1]; n++)
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
    MPI_Allreduce(MPI_IN_PLACE,&warpnzall,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);
    int *warpnztots = new int[numproc];
    MPI_Allgather(&warpnztot,1,MPI_INT,warpnztots,1,MPI_INT,MPI_COMM_WORLD);
    int warpnztotmin = warpnztots[0];
    int warpnztotmax = warpnztots[0];
    for(int p = 0; p < numproc; p++){
      if(warpnztotmin>warpnztots[p])warpnztotmin=warpnztots[p];
      if(warpnztotmax<warpnztots[p])warpnztotmax=warpnztots[p];
    }
    if(myid==0){
      printf("WARP ELL NZ: %ld (%f GB) OVERHEAD: %f EFFICIENCY: %f\n",warpnzall*(long)WARPSIZE,warpnzall*(double)WARPSIZE*(sizeof(MATPREC)+sizeof(unsigned short))/1.0e9,warpnzall*(double)WARPSIZE/proj_rownzall,warpnzall*(double)WARPSIZE*0.75/proj_rownzall);
      for(int p = 0; p < numproc; p++)
        printf("proc %d: warpnztot: %d (%f GB)\n",p,warpnztots[p]*WARPSIZE,warpnztots[p]/1.0e9*WARPSIZE*(sizeof(MATPREC)+sizeof(unsigned short)));
      printf("warpnztotmin: %d warpnztotmax: %d imbalance: %f\n",warpnztotmin*WARPSIZE,warpnztotmax*WARPSIZE,warpnztotmax/((double)warpnzall/numproc));
    }
    delete[] warpnztots;
    unsigned short *warpindex = new unsigned short[warpnztot*WARPSIZE];
    bool *warpindextag = new bool[warpnztot*WARPSIZE];
    #pragma omp parallel for
    for(int n = 0; n < warpnztot*WARPSIZE; n++){
      warpindex[n] = 0;
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
          for(int n = rowdispl[m]; n < rowdispl[m+1]; n++)
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
          for(int n = rowdispl[m]; n < rowdispl[m+1]; n++){
            int buffloc = numint[rowindex[n]];
            int *count = indcount+blocksize*buffloc;
            int indloc = m%blocksize;
            int warp = ((buffdispl[block]+buffloc)*blocksize+indloc)/WARPSIZE;
            int ind = (warpdispl[warp]+count[indloc])*WARPSIZE+m%WARPSIZE;
            warpindex[ind] = numind[rowindex[n]];
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
    back_warpindex = warpindex;
    back_warpindextag = warpindextag;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("BLOCKING TIME: %e\n",MPI_Wtime()-time);
  time = MPI_Wtime();
  if(myid==0)printf("\nFILL PROJECTION MATRIX\n");
  {
    proj_warpvalue = new MATPREC[proj_warpnztot*WARPSIZE];
    #pragma omp parallel for
    for(int n = 0; n < proj_warpnztot*WARPSIZE; n++)
      proj_warpvalue[n] = 0.0;
    #pragma omp parallel for
    for(int block = 0; block < proj_numblocks; block++){
      for(int ray = block*proj_blocksize; ray < (block+1)*proj_blocksize && ray < raynumout; ray++){
        double rho = raycoorout[ray].real();
        double theta = raycoorout[ray].imag();
        int n = ray%proj_blocksize;
        for(int buff = proj_buffdispl[block]; buff < proj_buffdispl[block+1]; buff++){
          int warp = (buff*proj_blocksize+n)/WARPSIZE;
          for(int row = proj_warpdispl[warp]; row < proj_warpdispl[warp+1]; row++){
            int ind = row*WARPSIZE+n%WARPSIZE;
            if(proj_warpindextag[ind]){
              int mapind = proj_mapdispl[buff]+proj_warpindex[ind];
              int pixind = proj_buffmap[mapind];
              double domain[4];
              domain[0]=pixcoor[pixind].real()-pixsize/2;
              domain[1]=domain[0]+pixsize;
              domain[2]=pixcoor[pixind].imag()-pixsize/2;
              domain[3]=domain[2]+pixsize;
              double temp = 0;
              findlength(theta,rho,domain,&temp);
              proj_warpvalue[ind] = temp;
            }
          }
        }
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("TIME: %e\n",MPI_Wtime()-time);
  time = MPI_Wtime();
  if(myid==0)printf("FILL BACKPROJECTION MATRIX\n");
  {
    back_warpvalue = new MATPREC[back_warpnztot*WARPSIZE];
    #pragma omp parallel for
    for(int n = 0; n < back_warpnztot*WARPSIZE; n++)
      back_warpvalue[n] = 0.0;
    #pragma omp parallel for
    for(int block = 0; block < back_numblocks; block++){
      for(int pix = block*back_blocksize; pix < (block+1)*back_blocksize && pix < mynumpix; pix++){
        double domain[4];
        domain[0]=pixcoor[pix].real()-pixsize/2;
        domain[1]=domain[0]+pixsize;
        domain[2]=pixcoor[pix].imag()-pixsize/2;
        domain[3]=domain[2]+pixsize;
        int n = pix%back_blocksize;
        for(int buff = back_buffdispl[block]; buff < back_buffdispl[block+1]; buff++){
          int warp = (buff*back_blocksize+n)/WARPSIZE;
          for(int row = back_warpdispl[warp]; row < back_warpdispl[warp+1]; row++){
            int ind = row*WARPSIZE+n%WARPSIZE;
            if(back_warpindextag[ind]){
              int mapind = back_mapdispl[buff]+back_warpindex[ind];
              int rayind = back_buffmap[mapind];
              double rho = raycoorout[rayind].real();
              double theta = raycoorout[rayind].imag();
              double temp = 0;
              findlength(theta,rho,domain,&temp);
              back_warpvalue[ind] = temp;
            }
          }
        }
      }
    }
  }
  delete[] raycoorout;
  delete[] raycoorinc;
  MPI_Barrier(MPI_COMM_WORLD);
  if(myid==0)printf("TIME: %e\n",MPI_Wtime()-time);
}

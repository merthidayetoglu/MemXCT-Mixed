#include <mpi.h>
#include <stdio.h>
#include <cmath>
#include <complex>
#include <limits>
#include <omp.h>
#include "mma.h"

#define OVERLAP //OVERLAP COMMUNICATIONS ON
#define FFACTOR 20 //FUSING FACTOR
#define WARPSIZE 32 //WARPSIZE
#define MATPREC half //MATRIX PRECISION
#define VECPREC half //VECTOR PRECISION
#define COMMPREC half //COMMUNICATION PRECISION
#define MATRIX //MATRIX STRUCTURE ON
#define MIXED //MIXED PRECISION ON

struct matrix{
  unsigned short ind;
  MATPREC val;
};

using namespace std;

void preproc();
void reducemap();
void communications();

void findnumpix(double,double,double*,int*);
void findpixind(double,double,double*,int*,int,int*);
void findlength(double,double,double*,double*);

void project(double*,double*,double,int);
void backproject(double*,double*,double,int);

int encode(unsigned short, unsigned short);
int xy2d (int n, int x, int y);
void d2xy(int n, int d, int *x, int *y);

void setup_gpu(double**,double**,double**,double**,double**,double**,double**);
double max_kernel(double*,int);
double dot_kernel(double*,double*,int);
void copyD2D_kernel(double*,double*,int);
void copyD2H_kernel(double*,double*,int);
void copyH2D_kernel(double*,double*,int);
void saxpy_kernel(double*,double*,double,double*,int);
void scale_kernel(double*,double,int);
void init_kernel(double*,int);

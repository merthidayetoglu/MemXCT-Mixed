#include <mpi.h>
#include <stdio.h>
#include <cmath>
#include <complex>
#include <limits>
#include <omp.h>
#include "mma.h"

#define FFACTOR 1 //FUSING FACTOR
#define WARPSIZE 32 //WARPSIZE
#define MATPREC half //MATRIX PRECISION
#define VECPREC half //VECTOR PRECISION

using namespace std;

void preproc();
void reducemap();
void communications();

void findnumpix(double,double,double*,int*);
void findpixind(double,double,double*,int*,int,int*);
void findlength(double,double,double*,double*);

void projection(double*,double*);
void backproject(double*,double*);

int encode(unsigned short, unsigned short);
int xy2d (int n, int x, int y);
void d2xy(int n, int d, int *x, int *y);

void setup_gpu(double**,double**,double**,double**,double**,double**);
double norm_kernel(double*,int);
double max_kernel(double*,int);
double dot_kernel(double*,double*,int);
void copy_kernel(double*,double*,int);
void saxpy_kernel(double*,double*,double,double*,int);
void scale_kernel(double*,double,int);

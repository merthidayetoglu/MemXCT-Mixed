#include "vars.h"

extern double raylength;
extern double pixsize;
extern int numx;
extern int numy;
extern int spatsize;

void findlength(double theta, double rho, double *d, double *length){
  //RAY'S VECTOR REPRESENTAION
  double x = rho*cos(theta)+0.5*raylength*sin(theta);
  double y = rho*sin(theta)-0.5*raylength*cos(theta);
  double dx = -raylength*sin(theta);
  double dy = +raylength*cos(theta);
  //TOP LEVEL
  double p[4] = {-dx,dx,-dy,dy};
  double q[4] = {x-d[0],d[1]-x,y-d[2],d[3]-y};
  double u1 = 0;
  double u2 = 1;
  bool pass = true;
  int inid = 0;
  for(int k = 0; k < 4; k++)
    if(p[k] == 0){
      if(q[k] < 0){
        pass = false;
        break;
      }
    }else{
      double t = q[k]/p[k];
      if(p[k] < 0 && u1 < t){
        u1 = t;
        inid = k;
      }
      else if(p[k] > 0 && u2 > t)
        u2 = t;
    }
  if(u1 > u2 || u1 > 1 || u1 < 0) pass = false;
  if(pass)if(u2>u1)*length = *length+(u2-u1)*raylength;
}

void findnumpix(double theta, double rho, double *d, int *numpix){
  //RAY'S VECTOR REPRESENTAION
  double x = rho*cos(theta)+0.5*raylength*sin(theta);
  double y = rho*sin(theta)-0.5*raylength*cos(theta);
  double dx = -raylength*sin(theta);
  double dy = +raylength*cos(theta);
  //TOP LEVEL
  double p[4] = {-dx,dx,-dy,dy};
  double q[4] = {x-d[0],d[1]-x,y-d[2],d[3]-y};
  double u1 = 0;
  double u2 = 1;
  bool pass = true;
  int inid = 0;
  for(int k = 0; k < 4; k++)
    if(p[k] == 0){
      if(q[k] < 0){
        pass = false;
        break;
      }
    }else{
      double t = q[k]/p[k];
      if(p[k] < 0 && u1 < t){
        u1 = t;
        inid = k;
      }
      else if(p[k] > 0 && u2 > t)
        u2 = t;
    }
  if(u1 > u2 || u1 > 1 || u1 < 0) pass = false;
  //IF RAY COLLIDES WITH DOMAIN
  if(pass){
    //FIND THE INITIAL PIXEL
    int init = 0;
    int initx = 0;
    int inity = 0;
    if(inid == 0){ //LEFT
      initx = 0;
      inity = (int)((y+u1*dy-d[2])/pixsize);
    }
    if(inid == 1){ //RIGHT
      initx = (int)((d[1]-d[0]-pixsize/2)/pixsize);
      inity = (int)((y+u1*dy-d[2])/pixsize);
    }
    if(inid == 2){ //BOTTOM
      initx = (int)((x+u1*dx-d[0])/pixsize);
      inity = 0;
    }
    if(inid == 3){ //TOP
      initx = (int)((x+u1*dx-d[0])/pixsize);
      inity = (int)((d[3]-d[2]-pixsize/2)/pixsize);
    }
    double px = d[0]+initx*pixsize+pixsize/2;
    double py = d[2]+inity*pixsize+pixsize/2;
    //TRACE RAY WHILE IT IS IN THE DOMAIN
    while(px > d[0] && px < d[1] && py < d[3] && py > d[2]){
      int exid = 0;
      q[0] = x-(px-pixsize/2);
      q[1] = (px+pixsize/2)-x;
      q[2] = y-(py-pixsize/2);
      q[3] = (py+pixsize/2)-y;
      u1 = 0;
      u2 = 1;
      for(int k = 0; k < 4; k++){
        double t = q[k]/p[k];
        if(p[k] < 0 && u1 < t)
          u1 = t;
        else if(p[k] > 0 && u2 > t){
          u2 = t;
          exid = k;
        }
      }
      //INCREMENT NUMBER OF PIXELS
      if(u2 > u1)
        *numpix = *numpix + 1;
      //FIND NEXT PIXEL
      if(exid == 0){
        initx = initx-1;
        px = px - pixsize;
      }
      if(exid == 1){
        initx = initx+1;
        px = px + pixsize;
      }
      if(exid == 2){
        inity = inity-1;
        py = py - pixsize;
      }
      if(exid == 3){
        inity = inity+1;
        py = py + pixsize;
      }
    }
  }
}

void findpixind(double theta, double rho, double *d, int *numpix, int offset, int *pixind){
  //RAY'S VECTOR REPRESENTAION
  double x = rho*cos(theta)+0.5*raylength*sin(theta);
  double y = rho*sin(theta)-0.5*raylength*cos(theta);
  double dx = -raylength*sin(theta);
  double dy = +raylength*cos(theta);
  //TOP LEVEL
  double p[4] = {-dx,dx,-dy,dy};
  double q[4] = {x-d[0],d[1]-x,y-d[2],d[3]-y};
  double u1 = 0;
  double u2 = 1;
  bool pass = true;
  int inid = 0;
  for(int k = 0; k < 4; k++)
    if(p[k] == 0){
      if(q[k] < 0){
        pass = false;
        break;
      }
    }else{
      double t = q[k]/p[k];
      if(p[k] < 0 && u1 < t){
        u1 = t;
        inid = k;
      }
      else if(p[k] > 0 && u2 > t)
        u2 = t;
    }
  if(u1 > u2 || u1 > 1 || u1 < 0) pass = false;
  //IF RAY COLLIDES WITH DOMAIN
  if(pass){
    //FIND THE INITIAL PIXEL
    int init = 0;
    int initx = 0;
    int inity = 0;
    if(inid == 0){ //LEFT
      initx = 0;
      inity = (int)((y+u1*dy-d[2])/pixsize);
    }
    if(inid == 1){ //RIGHT
      initx = (int)((d[1]-d[0]-pixsize/2)/pixsize);
      inity = (int)((y+u1*dy-d[2])/pixsize);
    }
    if(inid == 2){ //BOTTOM
      initx = (int)((x+u1*dx-d[0])/pixsize);
      inity = 0;
    }
    if(inid == 3){ //TOP
      initx = (int)((x+u1*dx-d[0])/pixsize);
      inity = (int)((d[3]-d[2]-pixsize/2)/pixsize);
    }
    double px = d[0]+initx*pixsize+pixsize/2;
    double py = d[2]+inity*pixsize+pixsize/2;
    //TRACE RAY WHILE IT IS IN THE DOMAIN
    while(px > d[0] && px < d[1] && py < d[3] && py > d[2]){
      int exid = 0;
      q[0] = x-(px-pixsize/2);
      q[1] = (px+pixsize/2)-x;
      q[2] = y-(py-pixsize/2);
      q[3] = (py+pixsize/2)-y;
      u1 = 0;
      u2 = 1;
      for(int k = 0; k < 4; k++){
        double t = q[k]/p[k];
        if(p[k] < 0 && u1 < t)
          u1 = t;
        else if(p[k] > 0 && u2 > t){
          u2 = t;
          exid = k;
        }
      }
      //ADD CONTRIBUTION FROM CURRENT PIXEL
      int  z = xy2d(spatsize,initx,inity);
      if(u2 > u1){
        pixind[*numpix] = offset+z;
        //pixlen[*numpix] = (u2-u1)*raylength;
        *numpix = *numpix + 1;
      }
      //FIND NEXT PIXEL
      if(exid == 0){
        initx = initx-1;
        px = px - pixsize;
      }
      if(exid == 1){
        initx = initx+1;
        px = px + pixsize;
      }
      if(exid == 2){
        inity = inity-1;
        py = py - pixsize;
      }
      if(exid == 3){
        inity = inity+1;
        py = py + pixsize;
      }
    }
  }
}


//rotate/flip a quadrant appropriately
void rot(int n, int *x, int *y, int rx, int ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n-1 - *x;
            *y = n-1 - *y;
        }

        //Swap x and y
        int t  = *x;
        *x = *y;
        *y = t;
    }
}
//convert (x,y) to d
int xy2d (int n, int x, int y) {
    int rx, ry, s, d=0;
    for (s=n/2; s>0; s/=2) {
        rx = (x & s) > 0;
        ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rot(s, &x, &y, rx, ry);
    }
    return d;
}
//convert d to (x,y)
void d2xy(int n, int d, int *x, int *y) {
    int rx, ry, s, t=d;
    *x = *y = 0;
    for (s=1; s<n; s*=2) {
        rx = 1 & (t/2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}

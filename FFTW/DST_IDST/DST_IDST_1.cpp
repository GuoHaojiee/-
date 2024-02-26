#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fftw3.h>

using namespace std;
/*
u(0) = u(1) = 0
u(x) = sin(x) 
x-(0, L) L = 2pi
M = 100  x_j = j*L/M  
*/ 

int main(){
    int M = 32;
    double L = 2* M_PI;
    double m = (M/2 - 1);
    double *u_j = (double*) fftw_malloc(sizeof(double) * M);
    double *u_j1 = (double*) fftw_malloc(sizeof(double) * M);
    double *u_j2 = (double*) fftw_malloc(sizeof(double) * M);
    double *u_k = (double*) fftw_malloc(sizeof(double) * m);

    for(int j = 0; j < M; j++){
        u_j[j] = sin((j) * L / M);   
    }
    
    for(int j = 0; j < m; j++){
        u_j1[j] = u_j[j+1];   
    }

    fftw_plan p1 = fftw_plan_r2r_1d(m, u_j1 , u_k, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(p1);
    for(int j = 0; j < (M/2 - 1); j++){
       u_k[j] /= M;   
       //cout << u_k[j] << " " << endl;
    }
    fftw_plan p2 = fftw_plan_r2r_1d(m, u_k, u_j2,  FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(p2);

    for(int j = m; j > 0 ; --j){
        u_j2[j] = u_j2[j-1];   
    }
    u_j2[0] = 0;
    
    for(int j = 0; j <= (M/2 - 1); j++){
        u_j2[M - j] = - u_j2[j];
    }
    
    // считать норму
    double err = 0.0;
    for (int j = 0; j < M; j++) {
        cout << u_j[j]<< " "<< " "<< u_j2[j]<< endl;
        double err_ = u_j[j] - u_j2[j];
        err += err_ * err_;
    }
    err = sqrt(err *L / M);
    cout << "||u_j - u_j2||_2 = " << err << endl;
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_free(u_j1);
    fftw_free(u_j2);
    fftw_free(u_k);
}


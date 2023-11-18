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
    int M = 5000;
    double L = 2 * M_PI;
    
    double *u_j1 = (double*) fftw_malloc(sizeof(double) * M);
    double *u_j2 = (double*) fftw_malloc(sizeof(double) * M);
    double *u_k = (double*) fftw_malloc(sizeof(double) * M);

    for(int j = 0; j < M; j++){
        u_j1[j] = sin(j * L / M);   
    }

    fftw_plan p1 = fftw_plan_r2r_1d(M, u_j1 , u_k, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(p1);
    for(int j = 0; j < M; j++){
       u_k[j] /= (2*(M+1));   
    }
    fftw_plan p2 = fftw_plan_r2r_1d(M, u_k, u_j2,  FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(p2);
    // считать норму
    double err = 0.0;
    for (int j = 0; j < M; j++) {
        //cout << u_j1[j]<< " "<< u_k[j] << " "<< u_j2[j]<< endl;
        double err_ = u_j1[j] - u_j2[j];
        err += err_ * err_;
    }
    err = sqrt(err *L / M);
    cout << "||u_j1 - u_j2||_2 = " << err << endl;
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_free(u_j1);
    fftw_free(u_j2);
    fftw_free(u_k);
}

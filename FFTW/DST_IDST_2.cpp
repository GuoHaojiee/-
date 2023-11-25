#include <stdio.h>
#include <iostream>
#include <math.h>
#include <fftw3.h>
#include <complex.h> 
/*
u(x) = sin(x)
x-(0, L) L = 4pi
N = 100  x_j = j*L/N  
*/ 
using namespace std;

int main(){
    int N = 32;
    double L = 2 * M_PI;
    int n = (N/2 -1);

    double *u_j = (double*) fftw_malloc(sizeof(double) * N);
    double *u_j1 = (double*) fftw_malloc(sizeof(double) * N);
    double *u_j2 = (double*) fftw_malloc(sizeof(double) * N);
    double *d2u_dx2 = (double*) fftw_malloc(sizeof(double) * N);
    double *u_k2 = (double*) fftw_malloc(sizeof(double) * N);
    double *u_k = (double*) fftw_malloc(sizeof(double) * n);

    for(int j = 0; j < N; j++){
        u_j[j] = sin(j * L / N);
        d2u_dx2[j]= -sin(j * L / N);  
        //cout << u_j1[j] << endl;
    }
    
    for(int j = 0; j < n; j++){
        u_j1[j] = u_j[j+1];
    }

    fftw_plan p1 = fftw_plan_r2r_1d(n , u_j1 , u_k, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(p1);
    
    for(int j = 0; j < (N/2-1); j++){
       u_k[j] /= N;
       //cout << u_k[j] << endl;
    }
    //cout << endl;
    
    for (int k = 0; k < n; k++) { 
        cout << u_k[k] << " ";
        u_k[k] = -(k+1)*(k+1)*u_k[k];
        cout << u_k[k] << endl;
    }

    fftw_plan p2 = fftw_plan_r2r_1d(n, u_k, u_j2,  FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(p2);

    for(int j = n; j > 0 ; --j){
        u_j2[j] = u_j2[j-1];
    }
    u_j2[0] = 0;

    for(int j = 0; j <= n; j++){
        u_j2[N - j] = - u_j2[j];
    }


    for(int k = 0; k < N; k++){
        cout << d2u_dx2[k] << " " << u_j2[k] << endl;
    }
    
    // считать норму ||u_j2 - du_dx||_2 
    double err = 0.0;
    for (int j = 0; j < N; j++) {
        double err_ = u_j2[j] - d2u_dx2[j];
        err += err_ * err_;
    }
    err = sqrt(err);
    cout << "||u_j2 - d2u/dx2||2 = " << err << endl;
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_free(u_j1);
    fftw_free(u_j2);
    fftw_free(d2u_dx2);
    fftw_free(u_k);
}

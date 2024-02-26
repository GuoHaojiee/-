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

double x_j(int j){
    return j*4* M_PI / 100;
}

int main(){
    int N = 100;
    double L = 4 * M_PI;
    
    double *u_j1 = (double*) fftw_malloc(sizeof(double) * N);
    double *u_j2 = (double*) fftw_malloc(sizeof(double) * N);
    double *du_dx = (double*) fftw_malloc(sizeof(double) * N);
    fftw_complex *u_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    for(int j = 0; j < N; j++){
        u_j1[j] = sin(x_j(j));
        du_dx[j]= cos(x_j(j));  
    }

    fftw_plan p1 = fftw_plan_dft_r2c_1d(N, u_j1 , u_k , FFTW_ESTIMATE);
    fftw_execute(p1);

    for (int k = 0; k < N; k++) {  
        double real = u_k[k][1] * (2 *M_PI * k / (N * L)) * (-1);
        double imaginary = u_k[k][0] * (2 *M_PI * k / (N * L));
        u_k[k][0] = real;
        u_k[k][1] = imaginary;
    }

    fftw_plan p2 = fftw_plan_dft_c2r_1d(N, u_k, u_j2 , FFTW_ESTIMATE);
    fftw_execute(p2);
    printf("%lf\n", u_j1[0]);// u(x) = sin(x) = 0
    printf("%lf\n", u_j2[0]);// du/dx(x) = cos(x) = 1
    // считать норму ||u_j2 - du_dx||_2 
    double err = 0.0;
    for (int j = 0; j < N; j++) {
        double err_ = u_j2[j] - du_dx[j];
        err += err_ * err_;
    }
    err = sqrt(err);
    cout << "||u_j2 - du_dx||_2 = " << err << endl;
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_free(u_j1);
    fftw_free(u_j2);
    fftw_free(du_dx);
    fftw_free(u_k);
}

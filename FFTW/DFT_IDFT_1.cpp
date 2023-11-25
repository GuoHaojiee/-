#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fftw3.h>

using namespace std;
/*
u(x) = sin(x)
x-(0, L) L = 4pi
N = 100  x_j = j*L/N  
*/ 

double x_j(int j){
    return j*4* M_PI / 100;
}

int main(){
    int N = 100;
    double L = 4 * M_PI;
    
    double *u_j1 = (double*) fftw_malloc(sizeof(double) * N);
    double *u_j2 = (double*) fftw_malloc(sizeof(double) * N);
    fftw_complex *u_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

    for(int j = 0; j < N; j++){
        u_j1[j] = sin(x_j(j));   
    }

    fftw_plan p1 = fftw_plan_dft_r2c_1d(N, u_j1 , u_k , FFTW_ESTIMATE);
    fftw_execute(p1);

    for (int k = 0; k < N; k++) {
        u_k[k][0] /= N;
        u_k[k][1] /= N;
    }

    fftw_plan p2 = fftw_plan_dft_c2r_1d(N, u_k, u_j2 , FFTW_ESTIMATE);
    fftw_execute(p2);
    // считать норму
    double err = 0.0;
    for (int j = 0; j < N; j++) {
        double err_ = u_j1[j] - u_j2[j];
        err += err_ * err_;
    }
    err = sqrt(err);
    cout << "||u_j1 - u_j2||_2 = " << err << endl;
    fftw_execute(p2);
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_free(u_j1);
    fftw_free(u_j2);
    fftw_free(u_k);
}
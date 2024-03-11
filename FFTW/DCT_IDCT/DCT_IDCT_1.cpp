/*---------------------
    DCT - IDCT шаг1: просто преобразование DCT - IDCT
  ---------------------  

    du/dx(0) = du/dx(1) = 0
    u(x) = cos(x) 
    x-(0, L) L = 2pi
*/ 

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <fftw3.h>

using namespace std;

int main(){
    int M = 8;
    double L = M_PI;
    int m = (M/2 + 1);

    double *u_j = (double*) fftw_malloc(sizeof(double) * M);
    double *u_j1 = (double*) fftw_malloc(sizeof(double) * M);
    double *u_j2 = (double*) fftw_malloc(sizeof(double) * M);
    double *u_k =  (double*) fftw_malloc(sizeof(double) * M);

    for(int j = 0; j < M; j++){
        u_j[j] = cos(j * L / M);   
    }


    fftw_plan p1 = fftw_plan_r2r_1d(M, u_j , u_k, FFTW_REDFT00, FFTW_ESTIMATE);
    fftw_execute(p1);

    for (int k = 0; k < M; k++) {
        u_k[k] /=  2*(M-1);
        cout << u_k[k] << endl;
    }

    fftw_plan p2 = fftw_plan_r2r_1d(M, u_k, u_j2,  FFTW_REDFT00, FFTW_ESTIMATE);
    fftw_execute(p2);

    // считать норму
    double err = 0.0;
    for (int j = 0; j < M; j++) {
        double err_ = u_j[j] - u_j2[j];
        err += err_ * err_;
    }
    err = sqrt(err * L / M);
    cout << "||u_j - u_j2||_2 = " << err << endl;
    fftw_execute(p2);
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_free(u_j1);
    fftw_free(u_j2);
    fftw_free(u_k);
}

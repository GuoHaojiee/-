/*
    Для граничных условий 2
    То есть du/dx(t, 0) = du/dx(t, L) = 0
    u(t, x) = (t + 1)cosx
*/

#include <iostream>
#include <cmath>
#include <fftw3.h>
#include <iomanip>
using namespace std;

double u_f(double t, double x) {
    return (t + 1 ) * cos(x);
}

double fi(double x) {
    return cos(x);
}

double f_(double t, double x) { 
    return  cos(x) + cos(x)*(t+1);
}

int main() {
    double L = 2 * M_PI;
    double T = 1;
    int M = 64, N = 1000;
    //cout << "Please input M , N : " << endl;
    int m = (M/2 + 1);

    double h = L / M;
    double tau = T / N;

    double* u = (double*) fftw_malloc(sizeof(double) * (M));
    double* u_1 = (double*) fftw_malloc(sizeof(double) * (M));
    double* u_2 = (double*) fftw_malloc(sizeof(double) * (M));
    double* f = (double*) fftw_malloc(sizeof(double) * (M));
    double* u_k_n = (double*)fftw_malloc(sizeof(double) * (m));
    double* u_k_n1 = (double*)fftw_malloc(sizeof(double) * (m));
    double* f_k = (double*)fftw_malloc(sizeof(double) * (m));
    
    for (int i = 0; i < M; ++i) {
        u[i] = fi(i * h);
    }

    for(int j = 0; j < m; j++){
        u_1[j] = fi(j * h);
    }

    fftw_plan p1 = fftw_plan_r2r_1d(m, u_1, u_k_n, FFTW_REDFT00, FFTW_ESTIMATE);
    fftw_execute(p1);
    for (int k = 0; k < m; k++) {
        u_k_n[k] /= M;
    }
    fftw_plan p2 = fftw_plan_r2r_1d(m, f , f_k, FFTW_REDFT00, FFTW_ESTIMATE);
    fftw_plan p3 = fftw_plan_r2r_1d(m, u_k_n1 , u_2, FFTW_REDFT00, FFTW_ESTIMATE);
    
    for (int n = 1; n <= N; n++) {
        for (int i = 0; i < m; i++) {
            f[i] = f_((n-1) * tau, ( i * h));
        }
        fftw_execute(p2);
        for (int k = 0; k < m; k++) {
            f_k[k] /= M;
        }
        for (int k = 0; k < m; k++) {
            u_k_n1[k] = u_k_n[k] - tau * k * k * u_k_n[k] + tau * f_k[k];
        }
        for (int k = 0; k < m; k++) {
            u_k_n[k] = u_k_n1[k];
        }
    }
    fftw_execute(p3);
    
    for (int j = 0; j < m; j++) {
        u_2[M - j] = u_2[j];
    }

    double error1, error2;
    double sum = 0;
    for(int i = 0; i < M; i++){
            //cout << u_f(1, i * h) << " " << u_2[i] << endl;
            sum += pow((u_f(1, i * h) - u_2[i]), 2);
    }
    error1 = sqrt(sum * h);
    double max = u_f(1, 0 * h) - u_2[0] ;
    double value;
    for(int i = 0; i < M; i++){
        value = abs(u_f(1, i * h) - u_2[i]);
        if (value > max)
            max = value;    
    }
    cout << "error = "<< error1 << endl;
    cout << "maxerr = "<<max << endl;
    
    fftw_destroy_plan(p3);
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p1);

    fftw_free(f_k);
    fftw_free(u_k_n1);
    fftw_free(u_k_n);
    fftw_free(f);
    fftw_free(u_1);
    fftw_free(u_2);
    fftw_free(u);
    return 0;
}

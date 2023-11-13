#include <iostream>
#include <cmath>
#include <fftw3.h>
#include <iomanip>
using namespace std;

double u_f(double t, double x) {
    return (t * t + 1) * sin(2 * M_PI * x);
}

double fi(double x) {
    return sin(2 * M_PI * x);
}

double pi_1(double t) {
    return 0;
}

double pi_2(double t) {
    return (t * t + 1) * sin(2 * M_PI);
}

double f_(double t, double x) {
    return 2 * t * sin(2 * M_PI * x) + 4 * M_PI * M_PI * (t * t + 1) * sin(2 * M_PI * x);
}

int main() {
    double L = 1;
    double T = 1;
    int M = 64, N = 32;
    cout << "Please input M , N : " << endl;

    double h = L / M;
    double tau = T / N;

    double* u = (double*) fftw_malloc(sizeof(double) * (M));
    double* f = (double*) fftw_malloc(sizeof(double) * (M));
    fftw_complex* u_k_n = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (M)/2+1);
    fftw_complex* u_k_n1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (M)/2+1);
    fftw_complex* f_k = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (M)/2+1);
    
    for (int i = 0; i < M; ++i) {
        u[i] = fi(i * h);
    }

    fftw_plan p1 = fftw_plan_dft_r2c_1d(M, u, u_k_n, FFTW_ESTIMATE);
    fftw_execute(p1);
    for (int k = 0; k < M/2+1; k++) {
        u_k_n[k][0] /= M;
        u_k_n[k][1] /= M;
    }
    fftw_plan p2 = fftw_plan_dft_r2c_1d(M, f, f_k, FFTW_ESTIMATE);
    fftw_plan p3 = fftw_plan_dft_c2r_1d(M, u_k_n1, u, FFTW_ESTIMATE);
    double alpha = 2 * M_PI / L;
    for (int n = 1; n <= N; n++) {
        for (int i = 0; i < M; ++i) {
            f[i] = f_((n-1) * tau, i * h);
        }
        fftw_execute(p2);
        for (int k = 0; k < M/2+1; k++) {
            f_k[k][0] /= M;
            f_k[k][1] /= M;
        }
        for (int k = 0; k < M/2+1; k++) {
            u_k_n1[k][0] = u_k_n[k][0] - (tau * alpha * alpha * k * k * u_k_n[k][0]) + tau * f_k[k][0];
            u_k_n1[k][1] = u_k_n[k][1] - (tau * alpha * alpha * k * k * u_k_n[k][1]) + tau * f_k[k][1];
        }
        if (n == 1){
            cout << "u_k_n: "<< endl;
            for(int k = 0; k < M/2+1; k++){   
                cout  << fixed << setprecision(30) <<u_k_n[k][0] <<" " << u_k_n[k][1] << endl;
            }
            cout << endl;
            cout << "f_k: "<< endl;
            for(int k = 0; k < M/2+1; k++){   
                cout<< fixed << setprecision(30)  << f_k[k][0] <<" " << f_k[k][1] << endl;
            }
            cout << endl;
            cout << "u_k_n1: "<< endl;
            for(int k = 0; k < M/2+1; k++){   
                cout << fixed << setprecision(30) << u_k_n1[k][0] <<" " << u_k_n1[k][1] << endl;
            }
            cout << endl;
        }
        for (int k = 0; k < M/2+1; k++) {
            u_k_n[k][0] = u_k_n1[k][0];
            u_k_n[k][1] = u_k_n1[k][1];
        }
        fftw_execute(p3);
        if (n == 1){
            for(int k = 0; k < M; k++){   
                cout<< fixed << setprecision(30)  << u[k] <<" " << u_f(tau * n, k * h) << endl;
            }
        }
    }
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p3);
    fftw_free(u_k_n);
    fftw_free(u_k_n1);
    fftw_free(f_k);
    fftw_free(f);
    fftw_free(u);
    return 0;
}

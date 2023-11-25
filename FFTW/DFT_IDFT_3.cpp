/*
    u(t,x) = (t + 1)exp(sin(x));
*/

#include <iostream>
#include <cmath>
#include <fftw3.h>
#include <iomanip>
using namespace std;

double u_f(double t, double x) {
    //return (t * t + 1) * cos(x);
    return (t + 1 ) * exp(sin(x));
}

double fi(double x) {
    //return cos(2 * M_PI * x);
    //return cos(x);
    return exp(sin(x));
}

double f_(double t, double x) {
    //return 2 * t * cos(2 * M_PI * x) + 4 * M_PI * M_PI * (t * t + 1) * cos(2 * M_PI * x);
    //return 2 * t * cos(x) + (t * t + 1) * cos(x);
    return exp(sin(x)) - (t + 1)*(exp(sin(x))* cos(x)*cos(x) -exp(sin(x))*sin(x));  
}

int main() {
    double L = 2 * M_PI;
    double T = 1;
    int M = 32, N = 1000;
    //cout << "Please input M , N : " << endl;

    double h = L / M;
    double tau = T / N;

    double* u = (double*) fftw_malloc(sizeof(double) * (M));
    double* f = (double*) fftw_malloc(sizeof(double) * (M));
    fftw_complex* u_k_n = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((M)/2+1));
    fftw_complex* u_k_n1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((M)/2+1));
    fftw_complex* f_k = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((M)/2+1));
    
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
        /*if (n == 2){
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
            cout << "u_k_n+1: "<< endl;
            for(int k = 0; k < M/2+1; k++){   
                cout << fixed << setprecision(30) << u_k_n1[k][0] <<" " << u_k_n1[k][1] << endl;
            }
            cout << endl;
        }*/
        for (int k = 0; k < M/2+1; k++) {
            u_k_n[k][0] = u_k_n1[k][0];
            u_k_n[k][1] = u_k_n1[k][1];
        }
        fftw_execute(p3);
        /*if (n == 200){
            for(int k = 0; k < M; k++){   
                cout<< fixed << setprecision(30)  << u[k] <<" " << u_f(tau * n, k * h) << endl;
            }
        }*/
    }
    double error1, error2;
    double sum = 0;
    for(int i = 0; i < M; i++){
            sum += pow((u_f(1, i * h) - u[i]), 2);
    }
    error1 = sqrt(sum * h);
    double max = u_f(1, 0 * h) - u[0] ;
    double value;
    for(int i = 0; i < M; i++){
        value = abs(u_f(1, i * h) - u[i]);
        if (value > max)
            max = value;    
    }
    cout << error1 << endl;
    cout << max << endl;
    
    fftw_destroy_plan(p3);
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p1);

    fftw_free(f_k);
    fftw_free(u_k_n1);
    fftw_free(u_k_n);
    fftw_free(f);
    fftw_free(u);
    return 0;
}

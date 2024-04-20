#include <iostream>
#include <cmath>
#include <fftw3.h>
#include <iomanip>
using namespace std;

double u_f(double t, double x) {
    return (t*t + 1) * cos(x);
    //return (t + 1 ) * exp(sin(x));
}

double fi(double x) {
    //return cos(2 * M_PI * x);
    return cos(x);
    //return exp(sin(x));
}

double F(double x) {
    return cos(x);
}

double f(double t){
    return t*t+ 2*t + 1;
}

void euler(fftw_complex* u_next, fftw_complex* F_complex, double alpha, int N, double tau, double L, int nx){
    for (int n = 1; n <= N; n++){
        for (int k = 0; k < (nx/2+1); k++) {
            u_next[k][0] = u_next[k][0] + tau *(-alpha * alpha *k*k*u_next[k][0] + F_complex[k][0] * f((n-1) *tau));
            u_next[k][1] = u_next[k][1] + tau *(-alpha * alpha *k*k*u_next[k][1] + F_complex[k][1] * f((n-1) *tau));
        }
    }
}

void rungeKutta(fftw_complex* u_next, fftw_complex* F_complex, double alpha, int N, double tau, double L, int nx) {
    for (int n = 1; n <= N; n++) {
        for (int k = 0; k < (nx/2+1); k++) {
            fftw_complex K1, K2, K3, K4;
            K1[0] = -alpha * alpha *k*k*u_next[k][0] + F_complex[k][0] * f((n-1) *tau);
            K1[1] = -alpha * alpha *k*k*u_next[k][1] + F_complex[k][1] * f((n-1) *tau);
            K2[0] = -alpha * alpha *k*k*(u_next[k][0] + tau / 2 * K1[0]) + F_complex[k][0] * f((n-1) * tau + tau / 2);
            K2[1] = -alpha * alpha *k*k*(u_next[k][1] + tau / 2 * K1[1]) + F_complex[k][1] * f((n-1) * tau + tau / 2);
            K3[0] = -alpha * alpha *k*k*(u_next[k][0] + tau / 2 * K2[0]) + F_complex[k][0] * f((n-1) * tau + tau / 2);
            K3[1] = -alpha * alpha *k*k*(u_next[k][1] + tau / 2 * K2[1]) + F_complex[k][1] * f((n-1) * tau + tau / 2);
            K4[0] = -alpha * alpha *k*k*(u_next[k][0] + tau * K3[0]) + F_complex[k][0] * f((n-1) * tau + tau);
            K4[1] = -alpha * alpha *k*k*(u_next[k][1] + tau * K3[1]) + F_complex[k][1] * f((n-1) * tau + tau);
            u_next[k][0] = u_next[k][0] + tau/6*(K1[0]+2*K2[0]+2*K3[0]+K4[0]);
            u_next[k][1] = u_next[k][1] + tau/6*(K1[1]+2*K2[1]+2*K3[1]+K4[1]);
        }
    }
}

void Teplo_2(fftw_complex* u_start, fftw_complex* u_out, fftw_complex* F_complex, double alpha, int N, double tau, double L, int nx, bool flag) {
    for (int i = 0; i < (nx/2+1); i++) {
        u_out[i][0] = u_start[i][0];
        u_out[i][1] = u_start[i][1];
    }

    if (flag == true)
        rungeKutta(u_out, F_complex, alpha, N, tau, L, nx);
    else
        euler(u_out, F_complex, alpha, N, tau, L, nx);

    for (int i = 0; i < (nx/2+1); i++) {
        u_out[i][0] /= nx;
        u_out[i][1] /= nx;
    }
}

double err_calculation(const double* n1, const  double* n2, int size){
    double err = 0.0;
    for (int i = 0; i < size; ++i) {
        double err_ = fabs(n1[i] - n2[i]);
        err += err_ * err_;
    }
    return  sqrt(err);
}

int main() {
    double L = 2 * M_PI;
    double T = 1;
    int M = 16, N = 64;
    double alpha = 2 * M_PI / L;
    double h = L / M;
    double tau = T / N;

    double* u_in = (double*) fftw_malloc(sizeof(double) * (M));
    double* F_in = (double*) fftw_malloc(sizeof(double) * (M));
    double* res = (double*) fftw_malloc(sizeof(double) * (M));
    double* out = (double*) fftw_malloc(sizeof(double) * (M));

    fftw_complex* u_start = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((M)/2+1));
    fftw_complex* u_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((M)/2+1));
    fftw_complex* F_complex = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((M)/2+1));
    
    for (int i = 0; i < M; ++i) {
        u_in[i] = fi(i * h);
        F_in[i] = F(i * h);
        res[i] =u_f(1, i*h) ;
    }

    fftw_plan p1 = fftw_plan_dft_r2c_1d(M, u_in, u_start, FFTW_ESTIMATE); //Forward transformation
    fftw_plan p3 = fftw_plan_dft_r2c_1d(M, F_in, F_complex, FFTW_ESTIMATE); //Forward transformation
    fftw_plan p2 = fftw_plan_dft_c2r_1d(M, u_out, out, FFTW_ESTIMATE);    //Backward transformation
    fftw_execute(p1);
    fftw_execute(p3);
    
    bool flag = true;
    Teplo_2(u_start,u_out, F_complex ,alpha,N,tau,L,M,flag);
    fftw_execute(p2);
    cout << "err_rungeKutta4 = " << err_calculation(res,out, M) << endl;
    
    flag = false;
    Teplo_2(u_start,u_out, F_complex ,alpha,N,tau,L,M,flag);
    fftw_execute(p2);
    cout << "err_euler = " << err_calculation(res,out, M) << endl;

    fftw_destroy_plan(p2);
    fftw_destroy_plan(p1);

    free(u_in);
    free(F_in);
    free(res);
    free(out);
    free(u_start);
    free(u_out);
    free(F_complex);
    return 0;
}

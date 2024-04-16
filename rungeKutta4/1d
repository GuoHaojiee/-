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

double f_(double t, double x) {
    //return 2 * t * cos(2 * M_PI * x) + 4 * M_PI * M_PI * (t * t + 1) * cos(2 * M_PI * x);
    return 2*t*cos(x) + (t*t + 1) * cos(x);
    //return exp(sin(x)) - (t + 1)*(exp(sin(x))* cos(x)*cos(x) -exp(sin(x))*sin(x));  
}

void calculateF(fftw_complex *f_1, fftw_complex *f_2, fftw_complex *f_3, int M, double tau, double h, int n)
{
    double *f1 = (double *)fftw_malloc(sizeof(double) * (M));
    double *f2 = (double *)fftw_malloc(sizeof(double) * (M));
    double *f3 = (double *)fftw_malloc(sizeof(double) * (M));
    fftw_plan plan_f1 = fftw_plan_dft_r2c_1d(M, f1, f_1, FFTW_ESTIMATE);
    fftw_plan plan_f2 = fftw_plan_dft_r2c_1d(M, f2, f_2, FFTW_ESTIMATE);
    fftw_plan plan_f3 = fftw_plan_dft_r2c_1d(M, f3, f_3, FFTW_ESTIMATE);
    for (int i = 0; i < M; ++i)
    {
        f1[i] = f_((n - 1) * tau, i * h);
        f2[i] = f_((n - 1) * tau + tau / 2, i * h);
        f3[i] = f_((n - 1) * tau + tau, i * h);
    }
    fftw_execute(plan_f1);
    fftw_execute(plan_f2);
    fftw_execute(plan_f3);
    fftw_destroy_plan(plan_f1);
    fftw_destroy_plan(plan_f2);
    fftw_destroy_plan(plan_f3);
    fftw_free(f1);
    fftw_free(f2);
    fftw_free(f3);
}

void euler(fftw_complex* u_prev, fftw_complex* u_next,  fftw_complex* f_1, 
                 double alpha, int N, double tau, double h, int M) {
    double *f1 = (double *)fftw_malloc(sizeof(double) * (M));
    fftw_complex* u_prev_ = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((M)/2+1));
    fftw_plan plan_f1 = fftw_plan_dft_r2c_1d(M, f1, f_1, FFTW_ESTIMATE);
    for (int k = 0; k < M/2+1; k++) {
            u_prev_[k][0] = u_prev[k][0];
            u_prev_[k][1] = u_prev[k][1];
    }

    for (int n = 1; n <= N; n++)
    {
        for (int i = 0; i < M; ++i)
        {
            f1[i] = f_((n - 1) * tau, i * h);
        }
        fftw_execute(plan_f1);
        for (int k = 0; k < M / 2 + 1; k++)
        {
            u_next[k][0] = u_prev_[k][0] + tau * (-alpha * alpha * k * k * u_prev_[k][0] + f_1[k][0]);
            u_next[k][1] = u_prev_[k][1] + tau * (-alpha * alpha * k * k * u_prev_[k][1] + f_1[k][1]);
        }
        for (int k = 0; k < M / 2 + 1; k++)
        {
            u_prev_[k][0] = u_next[k][0];
            u_prev_[k][1] = u_next[k][1];
        }
    }
    for (int k = 0; k < M / 2 + 1; k++)
    {
            u_next[k][0] /= M;
            u_next[k][1] /= M;
    }
    fftw_destroy_plan(plan_f1);
    fftw_free(u_prev_);
    fftw_free(f1);
}

void rungeKutta4(fftw_complex* u_prev, fftw_complex* u_next, fftw_complex* f_1, fftw_complex* f_2, fftw_complex* f_3, 
                 double alpha, int N, double tau, double h, int M) {
    fftw_complex *k1 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * ((M) / 2 + 1));
    fftw_complex *k2 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * ((M) / 2 + 1));
    fftw_complex *k3 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * ((M) / 2 + 1));
    fftw_complex *k4 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * ((M) / 2 + 1));
    fftw_complex* u_prev_ = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((M)/2+1));
    for (int k = 0; k < M/2+1; k++) {
            u_prev_[k][0] = u_prev[k][0];
            u_prev_[k][1] = u_prev[k][1];
    }
    
    for (int n = 1; n <= N; n++) {
        calculateF(f_1,f_2,f_3,M,tau,h,n);
        for (int k = 0; k < M/2+1; k++)
        {
            k1[k][0] = -alpha * alpha * k * k * u_prev_[k][0] + f_1[k][0];
            k1[k][1] = -alpha * alpha * k * k * u_prev_[k][1] + f_1[k][1];
            k2[k][0] = -alpha * alpha * k * k * (u_prev_[k][0] + tau / 2 * k1[k][0]) + f_2[k][0];
            k2[k][1] = -alpha * alpha * k * k * (u_prev_[k][1] + tau / 2 * k1[k][1]) + f_2[k][1];
            k3[k][0] = -alpha * alpha * k * k * (u_prev_[k][0] + tau / 2 * k2[k][0]) + f_2[k][0];
            k3[k][1] = -alpha * alpha * k * k * (u_prev_[k][1] + tau / 2 * k2[k][1]) + f_2[k][1];
            k4[k][0] = -alpha * alpha * k * k * (u_prev_[k][0] + tau * k3[k][0]) + f_3[k][0];
            k4[k][1] = -alpha * alpha * k * k * (u_prev_[k][1] + tau * k3[k][1]) + f_3[k][1];
        }

        for (int k = 0; k < M/2+1; k++) {
            u_next[k][0] = u_prev_[k][0] + tau/6*(k1[k][0]+2*k2[k][0]+2*k3[k][0]+k4[k][0]);
            u_next[k][1] = u_prev_[k][1] + tau/6*(k1[k][1]+2*k2[k][1]+2*k3[k][1]+k4[k][1]);
        }
        for (int k = 0; k < M/2+1; k++) {
            u_prev_[k][0] = u_next[k][0];
            u_prev_[k][1] = u_next[k][1];
        }
    }

    for (int k = 0; k < M / 2 + 1; k++)
    {
            u_next[k][0] /= M;
            u_next[k][1] /= M;
    }
    fftw_free(k1);
    fftw_free(k2);
    fftw_free(k3);
    fftw_free(k4);
    fftw_free(u_prev_);
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

    double h = L / M;
    double tau = T / N;

    double* u = (double*) fftw_malloc(sizeof(double) * (M));
    double* u_f1 = (double*) fftw_malloc(sizeof(double) * (M));
    double* u_res = (double*) fftw_malloc(sizeof(double) * (M));

    fftw_complex* u_prev = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((M)/2+1));
    fftw_complex* u_next = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((M)/2+1));
    fftw_complex* f_1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((M)/2+1));
    fftw_complex* f_2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((M)/2+1));
    fftw_complex* f_3 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * ((M)/2+1));
    
    for (int i = 0; i < M; ++i) {
        u[i] = fi(i * h);
        u_f1[i] =u_f(1, i*h) ;
    }

    fftw_plan p1 = fftw_plan_dft_r2c_1d(M, u, u_prev, FFTW_ESTIMATE);
    fftw_execute(p1);

    fftw_plan p2 = fftw_plan_dft_c2r_1d(M, u_next, u_res, FFTW_ESTIMATE);
    double alpha = 2 * M_PI / L;
    rungeKutta4(u_prev,u_next, f_1, f_2, f_3,alpha,N,tau,h,M);
    fftw_execute(p2);
    double err_rungeKutta4 = err_calculation(u_f1, u_res, M);
    cout << "err_rungeKutta4 = " << err_rungeKutta4 << endl;

    euler(u_prev,u_next, f_1,alpha,N,tau,h, M);
    fftw_execute(p2);
    double err_euler = err_calculation(u_f1, u_res, M);
    cout << "err_euler = " << err_euler << endl;
    
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p1);

    fftw_free(f_1);
    fftw_free(f_2);
    fftw_free(f_3);
    fftw_free(u_prev);
    fftw_free(u_next);
    fftw_free(u);
    return 0;
}

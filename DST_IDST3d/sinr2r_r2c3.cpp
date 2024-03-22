#include <iostream>
#include <fftw3.h>
#include <cmath>
#include <stdlib.h>
#include <string.h>

using namespace std;

double u(double x, double y, double z, double t) {
    return (t+1)*sin(x)*cos(y)*sin(2*M_PI*z);
}

double fi(double x, double y, double z) {
    return sin(x)*cos(y)*sin(2*M_PI*z);
}

double f_(double x, double y, double z, double t) {
    return ((4*M_PI*M_PI+2)*t + 4*M_PI*M_PI+3)*sin(x)*cos(y)*sin(2*M_PI*z);
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
    int N1 = 8; 
    int N2 = 8; 
    int N3 = 8;
    int n3 = N3/2-1;
    double L_x = 2*M_PI, L_y = 2*M_PI, L_z = 1;

    int Nt = 1000;
    double T = 1;
    double tau = T / Nt; 

    int istride, ostride;
    int idist, odist;
    int howmany;
    int rank;

    double *u_in = (double*) fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *f = (double*) fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *ino = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *u_res_fft = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *u_res = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    
    fftw_complex *u_prev = (fftw_complex*)fftw_malloc(N1*(N2/2+1)*n3 * sizeof(fftw_complex));
    fftw_complex *u_next = (fftw_complex*)fftw_malloc(N1*(N2/2+1)*n3 * sizeof(fftw_complex));
    fftw_complex *f_k = (fftw_complex*)fftw_malloc(N1*(N2/2+1)*n3 * sizeof(fftw_complex));
    
    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < N2; ++j) {
            for(int k = 1; k <= n3; ++k) {
                int index = (i*N2+j)*n3+k-1;
                u_in[index] = fi(i*L_x/N1, j*L_y/N2, k*L_z/N3);
                u_res[index] = u(i*L_x/N1, j*L_y/N2, k*L_z/N3, 1);
            }
        }
    }

    // Forward transformation R -> R z
    rank = 1;
    int n[] = {n3};
    howmany = N1*N2;
    istride = 1; ostride = 1;
    idist = n3;  odist = n3;
    int *inembed = n, *onembed = n;
    const fftw_r2r_kind kind[] = {FFTW_RODFT00};

    fftw_plan fplan_r2r_u_z = fftw_plan_many_r2r(rank, n, howmany,
                                              u_in, inembed, istride, idist,
                                              ino, onembed, ostride, odist,
                                              kind, FFTW_ESTIMATE);
    fftw_plan fplan_r2r_f_z = fftw_plan_many_r2r(rank, n, howmany,
                                              f, inembed, istride, idist,
                                              ino, onembed, ostride, odist,
                                              kind, FFTW_ESTIMATE);

    // Forward transformation R2 -> C2 x,y
    int nn[] = {N1, N2};
    int inembed2[] =  {N1, N2};
    int onembed2[] =  {N1, N2/2+1};
    istride = n3; ostride = n3;
    idist = 1; odist = 1;

    fftw_plan fplan_r2c_u_xy = fftw_plan_many_dft_r2c(2, nn, n3,
                                                ino, inembed2, istride, idist,
                                                u_prev, onembed2, ostride, odist,
                                                FFTW_ESTIMATE);
    fftw_plan fplan_r2c_f_xy = fftw_plan_many_dft_r2c(2, nn, n3,
                                                ino, inembed2, istride, idist,
                                                f_k, onembed2, ostride, odist,
                                                FFTW_ESTIMATE);                                  
    fftw_execute(fplan_r2r_u_z);
    fftw_execute(fplan_r2c_u_xy);

    //Normalization
    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < (N2/2+1); ++j) {
            for(int k = 0; k < n3; ++k) {
                int index = (i*(N2/2+1)+j)*n3+k;
                u_prev[index][0] /= N1*N2*N3;
                u_prev[index][1] /= N1*N2*N3;
            }
        }
    }  

    for (int n = 1; n <= Nt; n++) {
        for (int i = 0; i < N1; i++) {
            for (int j = 0; j < N2; j++) {
                for (int k = 1; k <= n3; k++) {
                    int index = (i*N2+j)*n3+k-1;
                    f[index] = f_(i*L_x/N1, j*L_y/N2, k*L_z/N3, (n-1)*T/Nt);
                }
            }
        }

        fftw_execute(fplan_r2r_f_z);
        fftw_execute(fplan_r2c_f_xy);

        for (int i = 0; i < N1; ++i){
            for (int j = 0; j < (N2 / 2 + 1); ++j){
                for (int k = 0; k < n3; ++k){
                    int index = (i * (N2 / 2 + 1) + j) * n3 + k;
                    f_k[index][0] /= N1 * N2 * N3;
                    f_k[index][1] /= N1 * N2 * N3;
                }
            }
        }

        for (int i = 0; i < N1; ++i){
            for (int j = 0; j < (N2 / 2 + 1); ++j){
                for (int k = 0; k < n3; ++k){
                    int index = (i * (N2 / 2 + 1) + j) * n3 + k;
                    double alpha = 2*M_PI/L_z;
                    int k_x = i <= N1/2 ? i : i - N1;
                    int k_y = j <= N2/2 ? j : j - N2;
                    u_next[index][0] = u_prev[index][0] + tau *((-1)* k_x * k_x * u_prev[index][0] + (-1)* k_y * k_y * u_prev[index][0] + -(k+1)*(k+1)*alpha*alpha * u_prev[index][0]) + tau * f_k[index][0];
                    u_next[index][1] = u_prev[index][1] + tau *((-1)* k_x * k_x * u_prev[index][1] + (-1)* k_y * k_y * u_prev[index][1] + -(k+1)*(k+1)*alpha*alpha * u_prev[index][1]) + tau * f_k[index][1];
                   }
            }
        }

        for (int i = 0; i < N1; ++i){
            for (int j = 0; j < (N2 / 2 + 1); ++j){
                for (int k = 0; k < n3; ++k){
                    int index = (i * (N2 / 2 + 1) + j) * n3 + k;
                    u_prev[index][0] = u_next[index][0];
                    u_prev[index][1] = u_next[index][1];
                }
            }
        }
    }
    
    // Backward transformation C2 -> R2 x,y
    inembed2[0] = N1; inembed2[1] = N2/2+1;
    onembed2[0] = N1; onembed2[1] = N2;
    istride = n3; ostride = n3;
    idist = 1; odist = 1;
    fftw_plan bplan_c2r = fftw_plan_many_dft_c2r(2, nn, n3,
                                                    u_next, inembed2, istride, idist,
                                                    ino, onembed2, ostride, odist,
                                                    FFTW_ESTIMATE);
        
    // Backward transformation R -> R z
    rank = 1;
    n[0] = n3;
    howmany = N1 * N2;
    istride = 1; ostride = 1;
    idist = n3; odist = n3;
    int *inembed3 = n, *onembed3 = n;
    fftw_plan bplan_r2r = fftw_plan_many_r2r(rank, n, howmany,
                                                      ino, inembed3, istride, idist,
                                                      u_res_fft, onembed3, ostride, odist,
                                                      kind, FFTW_ESTIMATE);
                                                      
    fftw_execute(bplan_c2r);
    fftw_execute(bplan_r2r);
    cout << "err u_fft - u = " << err_calculation(u_res_fft, u_res, N1*N2*n3) << endl;

    free(u_in);
    free(f);
    free(ino);
    free(u_res_fft);
    free(u_res);
    free(u_prev);
    free(u_next);
    free(f_k);
    fftw_destroy_plan(fplan_r2r_u_z);  
    fftw_destroy_plan(fplan_r2r_f_z); 
    fftw_destroy_plan(fplan_r2c_u_xy); 
    fftw_destroy_plan(fplan_r2c_f_xy); 
    fftw_destroy_plan(bplan_r2r); 
    fftw_destroy_plan(bplan_c2r); 
    return 0;
}
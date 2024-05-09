#include <iostream>
#include <fftw3.h>
#include <cmath>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <iomanip>

using namespace std;

double u(double x, double y, double z, double t) {
    return (t*t+1)*sin(x)*cos(y)*cos(2*M_PI*z);
}

double fi(double x, double y, double z) {
    return sin(x)*cos(y)*cos(2*M_PI*z);
}

double F(double x, double y, double z) {
    return sin(x)*cos(y)*cos(2*M_PI*z);
}

double f(double t){
    return ((4*M_PI*M_PI+2)*(t*t+1)+2*t);
}

void euler(fftw_complex* u_next, fftw_complex* F_complex, double alpha_x, double alpha_y,double alpha_z,int N, double tau, double L, int nx, int ny, int nz){
    for (int n = 1; n <= N; n++){
        #pragma omp parallel for proc_bind(spread)
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < (ny/2+1); j++) {
                for (int k = 0; k < (nz/2+1); k++) {
                    int index = (i * (ny/2+1) + j) * (nz/2+1) + k;
                    int k_x = i <= nx/2 ? i : i -nx;
                    int k_y = j <= ny/2 ? j : j -ny;
                    int k_z = k;
                    u_next[index][0] = u_next[index][0] + tau *(u_next[index][0] * (-alpha_x * alpha_x *k_x * k_x -alpha_y * alpha_y *k_y * k_y -alpha_z * alpha_z * k_z * k_z) + F_complex[index][0] * f((n-1) *tau));
                    u_next[index][1] = u_next[index][1] + tau *(u_next[index][1] * (-alpha_x * alpha_x *k_x * k_x -alpha_y * alpha_y *k_y * k_y -alpha_z * alpha_z * k_z * k_z) + F_complex[index][1] * f((n-1) *tau));
                }
            }
        }
    }
}

void rungeKutta(fftw_complex* u_next, fftw_complex* F_complex,  double alpha_x, double alpha_y,double alpha_z, int N, double tau, double L, int nx, int ny, int nz) {
    for (int n = 1; n <= N; n++) {
        #pragma omp parallel for proc_bind(spread) 
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j <(ny/2+1); j++) {
                for (int k = 0; k < (nz/2+1); k++) {
                    fftw_complex K1, K2, K3, K4;
                    int index = (i * (ny/2+1) + j) * (nz/2+1) + k;
                    int k_x = i <= nx/2 ? i : i -nx;
                    int k_y = j <= ny/2 ? j : j -ny;
                    int k_z = k;
                    K1[0] = u_next[index][0] * (-alpha_x * alpha_x *k_x * k_x -alpha_y * alpha_y *k_y * k_y -alpha_z * alpha_z * k_z * k_z) + F_complex[index][0] * f((n-1) *tau);
                    K1[1] = u_next[index][1] * (-alpha_x * alpha_x *k_x * k_x -alpha_y * alpha_y *k_y * k_y -alpha_z * alpha_z * k_z * k_z) + F_complex[index][1] * f((n-1) *tau);
                    K2[0] = (u_next[index][0] + tau / 2 * K1[0]) * (-alpha_x * alpha_x *k_x * k_x -alpha_y * alpha_y *k_y * k_y -alpha_z * alpha_z * k_z * k_z) + F_complex[index][0] * f((n-1) * tau + tau / 2);
                    K2[1] = (u_next[index][1] + tau / 2 * K1[1]) * (-alpha_x * alpha_x *k_x * k_x -alpha_y * alpha_y *k_y * k_y -alpha_z * alpha_z * k_z * k_z) + F_complex[index][1] * f((n-1) * tau + tau / 2);
                    K3[0] = (u_next[index][0] + tau / 2 * K2[0]) * (-alpha_x * alpha_x *k_x * k_x -alpha_y * alpha_y *k_y * k_y -alpha_z * alpha_z * k_z * k_z) + F_complex[index][0] * f((n-1) * tau + tau / 2);
                    K3[1] = (u_next[index][1] + tau / 2 * K2[1]) * (-alpha_x * alpha_x *k_x * k_x -alpha_y * alpha_y *k_y * k_y -alpha_z * alpha_z * k_z * k_z) + F_complex[index][1] * f((n-1) * tau + tau / 2);
                    K4[0] = (u_next[index][0] + tau * K3[0]) * (-alpha_x * alpha_x *k_x * k_x -alpha_y * alpha_y *k_y * k_y -alpha_z * alpha_z * k_z * k_z) + F_complex[index][0] * f((n-1) * tau + tau);
                    K4[1] = (u_next[index][1] + tau * K3[1]) * (-alpha_x * alpha_x *k_x * k_x -alpha_y * alpha_y *k_y * k_y -alpha_z * alpha_z * k_z * k_z) + F_complex[index][1] * f((n-1) * tau + tau);
                    u_next[index][0] = u_next[index][0] + tau/6*(K1[0]+2*K2[0]+2*K3[0]+K4[0]);
                    u_next[index][1] = u_next[index][1] + tau/6*(K1[1]+2*K2[1]+2*K3[1]+K4[1]);
                }
            }
        }
    }
}

void Teplo_2(fftw_complex* u_start, fftw_complex* u_out, fftw_complex* F_complex, double alpha_x, double alpha_y,double alpha_z, int N, double tau, double L, int nx, int ny, int nz, bool flag) {
    #pragma omp parallel for proc_bind(spread)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < (ny/2+1); j++) {  
            for (int k = 0; k < (nz/2+1); k++) {
                int index = (i * (ny/2+1) + j) * (nz/2+1) + k;
                u_out[index][0] = u_start[index][0];
                u_out[index][1] = u_start[index][1];
            }
        }
    }

    if (flag == true)
        rungeKutta(u_out, F_complex, alpha_x, alpha_y, alpha_z, N, tau, L, nx, ny, nz);
    else
        euler(u_out, F_complex, alpha_x, alpha_y, alpha_z, N, tau, L, nx, ny, nz);
    
    #pragma omp parallel for proc_bind(spread)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < (ny/2+1); j++) {
            for (int k = 0; k < (nz/2+1); k++) {
                int index = (i * (ny/2+1) + j) * (nz/2+1) + k;
                u_out[index][0] /= (nx * ny * nz);
                u_out[index][1] /= (nx * ny * nz);
            }
        }
    }
}

double err_calculation(const double* n1, const  double* n2, int size){
    double err = 0.0;
    #pragma omp parallel for proc_bind(spread)
    for (int i = 0; i < size; ++i) {
        double err_ = fabs(n1[i] - n2[i]);
        err += err_ * err_;
    }
    return  sqrt(err);
}

int main() {
    int N1 = 64; 
    int N2 = 64; 
    int N3 = 64;
    double L_x = 2*M_PI, L_y = 2*M_PI, L_z = 1;
    double alpha_x = 2 * M_PI / L_x;
    double alpha_y = 2 * M_PI / L_y;
    double alpha_z = 2 * M_PI / L_z;

    int Nt = 25000;
    double T = 1;
    double tau = T / Nt; 

    int max_threads = omp_get_max_threads();
    cout << "num threads = " << max_threads << endl;
    fftw_init_threads();
    fftw_plan_with_nthreads(max_threads);
    omp_set_num_threads(max_threads);

    int istride, ostride;
    int idist, odist;
    int howmany;
    int rank;

    double *u_in = (double*) fftw_malloc(sizeof(double) * N1 * N2 * (N3/2+1));
    double *F_in = (double*) fftw_malloc(sizeof(double) * N1 * N2 * (N3/2+1));
    double *ino = (double*)fftw_malloc(sizeof(double) * N1 * N2 * (N3/2+1));
    double *res = (double*)fftw_malloc(sizeof(double) * N1 * N2 * (N3/2+1));
    double *out = (double*)fftw_malloc(sizeof(double) * N1 * N2 * (N3/2+1));
    
    fftw_complex *u_start = (fftw_complex*)fftw_malloc(N1*(N2/2+1)*(N3/2+1) * sizeof(fftw_complex));
    fftw_complex *u_out = (fftw_complex*)fftw_malloc(N1*(N2/2+1)*(N3/2+1) * sizeof(fftw_complex));
    fftw_complex *F_complex = (fftw_complex*)fftw_malloc(N1*(N2/2+1)*(N3/2+1) * sizeof(fftw_complex));
    double start = omp_get_wtime();

    #pragma omp parallel for proc_bind(spread) 
    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < N2; ++j) {
            for(int k = 0; k < (N3/2+1); ++k) {
                int index = (i*N2+j)*(N3/2+1)+k;
                u_in[index] = fi(i*L_x/N1, j*L_y/N2, k*L_z/N3);
                F_in[index] = F(i*L_x/N1, j*L_y/N2, k*L_z/N3);
                out[index] = u(i*L_x/N1, j*L_y/N2, k*L_z/N3, 1);
            }
        }
    }

    // Forward transformation R -> R z
    rank = 1;
    int n[] = {(N3/2+1)};
    howmany = N1*N2;
    istride = 1; ostride = 1;
    idist = (N3/2+1);  odist = (N3/2+1);
    int *inembed = n, *onembed = n;
    const fftw_r2r_kind kind[] = {FFTW_REDFT00};

    fftw_plan fplan_r2r_u_z = fftw_plan_many_r2r(rank, n, howmany,
                                              u_in, inembed, istride, idist,
                                              ino, onembed, ostride, odist,
                                              kind, FFTW_ESTIMATE);
    fftw_plan fplan_r2r_F_z = fftw_plan_many_r2r(rank, n, howmany,
                                              F_in, inembed, istride, idist,
                                              ino, onembed, ostride, odist,
                                              kind, FFTW_ESTIMATE);

    // Forward transformation R2 -> C2 x,y
    int nn[] = {N1, N2};
    int inembed2[] =  {N1, N2};
    int onembed2[] =  {N1, N2/2+1};
    istride = (N3/2+1); ostride = (N3/2+1);
    idist = 1; odist = 1;

    fftw_plan fplan_r2c_u_xy = fftw_plan_many_dft_r2c(2, nn, (N3/2+1),
                                                ino, inembed2, istride, idist,
                                                u_start, onembed2, ostride, odist,
                                                FFTW_ESTIMATE);
    fftw_plan fplan_r2c_F_xy = fftw_plan_many_dft_r2c(2, nn, (N3/2+1),
                                                ino, inembed2, istride, idist,
                                                F_complex, onembed2, ostride, odist,
                                                FFTW_ESTIMATE);   

    // Backward transformation C2 -> R2 x,y
    inembed2[0] = N1; inembed2[1] = N2/2+1;
    onembed2[0] = N1; onembed2[1] = N2;
    istride = (N3/2+1); ostride = (N3/2+1);
    idist = 1; odist = 1;
    fftw_plan bplan_c2r = fftw_plan_many_dft_c2r(2, nn, (N3/2+1),
                                                    u_out, inembed2, istride, idist,
                                                    ino, onembed2, ostride, odist,
                                                    FFTW_ESTIMATE);
        
    // Backward transformation R -> R z
    rank = 1;
    n[0] = (N3/2+1);
    howmany = N1 * N2;
    istride = 1; ostride = 1;
    idist = (N3/2+1); odist = (N3/2+1);
    int *inembed3 = n, *onembed3 = n;
    fftw_plan bplan_r2r = fftw_plan_many_r2r(rank, n, howmany,
                                                      ino, inembed3, istride, idist,
                                                      res, onembed3, ostride, odist,
                                                      kind, FFTW_ESTIMATE);
                                                                                     
    fftw_execute(fplan_r2r_u_z);
    fftw_execute(fplan_r2c_u_xy);
    fftw_execute(fplan_r2r_F_z);
    fftw_execute(fplan_r2c_F_xy);
    
    bool flag = false;
    Teplo_2(u_start,u_out,F_complex, alpha_x, alpha_y, alpha_z, Nt, tau, L_x, N1, N2, N3, flag);  
    fftw_execute(bplan_c2r);
    fftw_execute(bplan_r2r);
    cout << "err_euler = " << err_calculation(res, out, N1*N2*(N3/2+1)) << endl;

    flag = true;
    Teplo_2(u_start,u_out,F_complex, alpha_x, alpha_y, alpha_z, Nt, tau, L_x, N1, N2, N3, flag);  
    fftw_execute(bplan_c2r);
    fftw_execute(bplan_r2r);
    cout << "err_rungeKutta4 = " << err_calculation(res, out, N1*N2*(N3/2+1)) << endl;

    double end = omp_get_wtime();
    cout << "time parallel = " << fixed << setprecision(2) << (end - start) << "s" << endl;

    fftw_cleanup_threads();
    free(u_in);
    free(F_in);
    free(ino);
    free(res);
    free(out);
    free(u_start);
    free(u_out);
    free(F_complex);
    fftw_destroy_plan(fplan_r2r_u_z);  
    fftw_destroy_plan(fplan_r2r_F_z); 
    fftw_destroy_plan(fplan_r2c_u_xy); 
    fftw_destroy_plan(fplan_r2c_F_xy); 
    fftw_destroy_plan(bplan_r2r); 
    fftw_destroy_plan(bplan_c2r); 
    return 0;
}
/*---------------------
    DFT - IDFT 3d шаг3: 3d Heat_equeation
  ---------------------  
    u(x,y,z,t) = (t + 1) * sin4x * cos5y * sin3z
    x, y ,z -(0, L) L = 2pi  t - (0, 1)
*/ 

#include <iostream>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <iomanip>
using namespace std;

double u(double x, double y, double z, double t) {
    return (t*t + 1 ) * sin(4 * x) * cos(5 * y) * sin(3  * z);
}

double fi(double x, double y, double z) {
    return sin(4 * x) * cos(5 * y) * sin(3  * z);
}

double F(double x, double y, double z) {
    return sin(4 * x) * cos(5 * y) * sin(3  * z);
}

double f(double t){
    return (50*t*t+2*t+50);
}

void euler(fftw_complex* u_next, fftw_complex* F_complex, double alpha, int N, double tau, double L, int nx, int ny, int nz){
    for (int n = 1; n <= N; n++){
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < (nz / 2 + 1); k++) {
                    int index = (i * ny + j) * (nz / 2 + 1) + k;
                    int k_x = i <= nx/2 ? i : i -nx;
                    int k_y = j <= ny/2 ? j : j -ny;
                    int k_z = k;
                    u_next[index][0] = u_next[index][0] + tau *(-alpha * alpha *u_next[index][0] * (k_x * k_x + k_y * k_y + k_z * k_z) + F_complex[index][0] * f((n-1) *tau));
                    u_next[index][1] = u_next[index][1] + tau *(-alpha * alpha *u_next[index][1] * (k_x * k_x + k_y * k_y + k_z * k_z) + F_complex[index][1] * f((n-1) *tau));
                }
            }
        }
    }
}

void rungeKutta(fftw_complex* u_next, fftw_complex* F_complex, double alpha, int N, double tau, double L, int nx, int ny, int nz) {
    for (int n = 1; n <= N; n++) {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < (nz / 2 + 1); k++) {
                    int index = (i * ny + j) * (nz / 2 + 1) + k;
                    int k_x = i <= nx/2 ? i : i -nx;
                    int k_y = j <= ny/2 ? j : j -ny;
                    int k_z = k;
                    fftw_complex K1, K2, K3, K4;
                    K1[0] = -alpha * alpha *u_next[index][0] * (k_x * k_x + k_y * k_y + k_z * k_z) + F_complex[index][0] * f((n-1) *tau);
                    K1[1] = -alpha * alpha *u_next[index][1] * (k_x * k_x + k_y * k_y + k_z * k_z) + F_complex[index][1] * f((n-1) *tau);
                    K2[0] = -alpha * alpha *(u_next[index][0] + tau / 2 * K1[0]) * (k_x * k_x + k_y * k_y + k_z * k_z) + F_complex[index][0] * f((n-1) * tau + tau / 2);
                    K2[1] = -alpha * alpha *(u_next[index][1] + tau / 2 * K1[1]) * (k_x * k_x + k_y * k_y + k_z * k_z) + F_complex[index][1] * f((n-1) * tau + tau / 2);
                    K3[0] = -alpha * alpha *(u_next[index][0] + tau / 2 * K2[0]) * (k_x * k_x + k_y * k_y + k_z * k_z) + F_complex[index][0] * f((n-1) * tau + tau / 2);
                    K3[1] = -alpha * alpha *(u_next[index][1] + tau / 2 * K2[1]) * (k_x * k_x + k_y * k_y + k_z * k_z) + F_complex[index][1] * f((n-1) * tau + tau / 2);
                    K4[0] = -alpha * alpha *(u_next[index][0] + tau * K3[0]) * (k_x * k_x + k_y * k_y + k_z * k_z) + F_complex[index][0] * f((n-1) * tau + tau);
                    K4[1] = -alpha * alpha *(u_next[index][1] + tau * K3[1]) * (k_x * k_x + k_y * k_y + k_z * k_z) + F_complex[index][1] * f((n-1) * tau + tau);
                    u_next[index][0] = u_next[index][0] + tau/6*(K1[0]+2*K2[0]+2*K3[0]+K4[0]);
                    u_next[index][1] = u_next[index][1] + tau/6*(K1[1]+2*K2[1]+2*K3[1]+K4[1]);
                }
            }
        }
    }
}

void Teplo_2(fftw_complex* u_start, fftw_complex* u_out, fftw_complex* F_complex, double alpha, int N, double tau, double L, int nx, int ny, int nz, bool flag) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {  
            for (int k = 0; k < (nz / 2 + 1); k++) {
                int index = (i * ny + j) * (nz / 2 + 1) + k;
                u_out[index][0] = u_start[index][0];
                u_out[index][1] = u_start[index][1];
            }
        }
    }

    if (flag == true)
        rungeKutta(u_out, F_complex, alpha, N, tau, L, nx, ny, nz);
    else
        euler(u_out, F_complex, alpha, N, tau, L, nx, ny, nz);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < (nz / 2 + 1); k++) {
                int index = (i * ny + j) * (nz / 2 + 1) + k;
                u_out[index][0] /= (nx * ny * nz);
                u_out[index][1] /= (nx * ny * nz);
            }
        }
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
    int nx = 16; 
    int ny = 16; 
    int nz = 16;
    int nt = 256;
    double T = 1;
    double L = 2 * M_PI;

    double tau = T / nt;  
    double alpha = 2 * M_PI / L;
    fftw_plan plan1, plan2, plan3;
    int i, j, k, index;

    double *u_in = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *F_in = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *out = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *res = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    fftw_complex *F_complex = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *u_start = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *u_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                u_in[(i*ny + j)*nz + k] = fi(i*L/nx, j*L/ny, k*L/nz);
                F_in[(i*ny + j)*nz + k] = F(i*L/nx, j*L/ny, k*L/nz);
                out[(i*ny + j)*nz + k] = u(i*L/nx, j*L/ny, k*L/nz, 1);
            }
        }
    }

    plan1 = fftw_plan_dft_r2c_3d(nx, ny, nz, u_in, u_start, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_r2c_3d(nx, ny, nz, F_in, F_complex, FFTW_ESTIMATE);
    plan3 = fftw_plan_dft_c2r_3d(nx, ny, nz, u_out, res, FFTW_ESTIMATE);
    fftw_execute(plan1);
    fftw_execute(plan2);

    bool flag = false; 
    Teplo_2(u_start,u_out,F_complex, alpha, nt, tau, L, nx, ny, nz, flag);
    fftw_execute(plan3);
    double err_euler = err_calculation(out, res, nx*ny*nz);
    cout << "err_euler = " << err_euler << endl;
    
    flag = true; 
    Teplo_2(u_start,u_out,F_complex, alpha, nt, tau, L, nx, ny, nz, flag);
    fftw_execute(plan3);
    double err_rungeKutta4 = err_calculation(out, res, nx*ny*nz);
    cout << "err_rungeKutta4 = " << err_rungeKutta4 << endl;

    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);
    fftw_destroy_plan(plan3);
    fftw_free(u_start);
    fftw_free(u_out);
    fftw_free(F_complex);
    fftw_free(u_in);
    fftw_free(F_in);
    fftw_free(res);
    fftw_free(out);
    return 0;
}

/*---------------------------------------------------------------------
    DST - IDST 3d шаг1: 3d Heat_equeation
    u(x,y,z) периодичное по х и по у, а по z граничные условия первого!
  ---------------------------------------------------------------------  
    u(x,y,z,t) = (1+t)cosx * cosy * sinz
    x, y ,z -(0, L) L = 2pi
*/ 
#include <iostream>
#include <fftw3.h>
#include <cmath>
#include <stdlib.h>

using namespace std;

double u(double x, double y, double z, double t) {
    return (t + 1 ) * cos(x) * cos(y) * sin(z);
}

double fi(double x, double y, double z) {
    return cos(x) * cos(y) * sin(z);
}

double f_(double x, double y, double z, double t) {
    return (3 * t + 4) * cos(x) * cos(y) * sin(z);
}

int main() {
    int Nx = 8; 
    int Ny = 8; 
    int Nz = 8;
    int Nt = 1000;
    int nx = (Nx/2+1);
    int ny = (Ny/2+1);
    int nz = (Nz/2-1);
    double L = 2 * M_PI;
    double T = 1;
    double tau = T / Nt;  
    fftw_plan plan1, plan2, plan3;
    int i, j, k, index;

    double *u_in = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    
    double * u = (double*) fftw_malloc(sizeof(double) * Nx * Ny * Nz);
    double * u_res = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    
    double *f = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *f_k = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *u_prev = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *u_next = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                u_in[(i*ny + j)*nz + k] = fi(i*L/Nx, j*L/Ny, (k+1)*L/Nz);
            }
        }
    }

    plan1 = fftw_plan_r2r_3d(nx, ny, nz, u_in, u_prev, FFTW_REDFT00, FFTW_REDFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    plan2 = fftw_plan_r2r_3d(nx, ny, nz, f, f_k, FFTW_REDFT00, FFTW_REDFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(plan1);
    
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                index = (i * ny + j) * nz + k;
                u_prev[index] /= Nx * Ny * Nz;
            }
        }
    }
    
    for (int n = 1; n <= Nt; n++) {
        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                for (k = 0; k < nz; k++) {
                    f[(i*ny + j)*nz + k] = f_(i*L/Nx, j*L/Ny, (k+1)*L/Nz, (n-1)*T/Nt);
                }
            }
        }

        fftw_execute(plan2);
        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                for (k = 0; k < nz; k++) {
                    index = (i * ny + j) * nz + k;
                    f_k[index] /= Nx * Ny * Nz;
                }
            }
        }
    
        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                for (k = 0; k < nz; k++) {
                    index = (i * ny + j) * nz + k;
                    double k_x = i;
                    double k_y = j;
                    double k_z = k+1;
                    u_next[index] = u_prev[index] + tau *((-1)* k_x * k_x * u_prev[index] + (-1)* k_y * k_y * u_prev[index] + (-1)* k_z * k_z * u_prev[index]) + tau * f_k[index];
                   }
            }
        }

        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                for (k = 0; k < nz; k++) {
                    index = (i * ny + j) * nz + k;
                    u_prev[index] = u_next[index];
                }
            }
        }
    }
    
    plan3 = fftw_plan_r2r_3d(nx, ny, nz, u_next, u_res, FFTW_REDFT00, FFTW_REDFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(plan3);

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = nz; k > 0; k--) {
                u[(i * Ny + j) * Nz + k] = u_res[(i * ny + j) * nz + k-1];
            }
            u[(i * Ny + j) * Nz] =  0;
            for(int k = 0; k < nz; k++){
                u[(i * Ny + j) * Nz + Nz - k -1] = -u_res[(i * ny + j) * nz + k];
            }
            u[(i * Ny + j) * Nz + nz+1] =  0;
        }
    }

    for (i = 0; i < Nx; i++) {
        for (j = 1; j < ny; j++) {
            for (k = 0; k < Nz; k++) {
                u[(i * Ny + Ny-j) * Nz + k] = u[(i * Ny + j) * Nz + k];
            }
        }
    }

    for (i = 1; i < nx; i++) {
        for (j = 0; j < Ny; j++) {
            for (k = 0; k < Nz; k++) {
                u[((Nx-i) * Ny + j) * Nz + k] = u[(i * Ny + j) * Nz + k];
            }
        }
    }

    double err = 0.0;
    for (i = 0; i < Nx; i++) {
        for (j = 0; j < Ny; j++) {
            for (k = 0; k < Nz; k++) {
                double res = (1 + 1) * cos(i * L / Nx) * cos(j * L / Ny) * sin(k * L / Nz);
                double err_ = fabs(u[(i * Ny + j) * Nz + k] - res);
                err += err_ * err_;
            }
        }  
    }
    err = sqrt(err);

    cout << "err = " << err << endl;
    
    fftw_destroy_plan(plan3);
    fftw_destroy_plan(plan2);
    fftw_destroy_plan(plan1);
    fftw_free(u_next);
    fftw_free(u_prev);
    fftw_free(f_k);
    fftw_free(f);
    fftw_free(u_res);
    fftw_free(u);
    fftw_free(u_in);
    return 0;
}
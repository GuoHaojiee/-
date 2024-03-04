/*---------------------------------------------------------------------
    DST - IDST 3d шаг1: просто преобразование DST - IDST 3d
    u(x,y,z) периодичное по х и по у, а по z граничные условия первого!
  ---------------------------------------------------------------------  
    u(x,y,z) = cosx * cosy * sinz
    x, y ,z -(0, L) L = 2pi
*/ 
#include <iostream>
#include <fftw3.h>
#include <cmath>
#include <stdlib.h>

using namespace std;

int main() {
    int Nx = 16; 
    int Ny = 16; 
    int Nz = 16;
    int nx = (Nx/2+1);
    int ny = (Ny/2+1);
    int nz = (Nz/2-1);
    double L = 2 * M_PI;
    fftw_plan plan1, plan2;
    int i, j, k, index;

    double *in = (double*) fftw_malloc(sizeof(double) * Nx * Ny * Nz);
    double *in1 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *out = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *res1 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *res2 = (double*) fftw_malloc(sizeof(double) * Nx * Ny * Nz);

    for (i = 0; i < Nx; i++) {
        for (j = 0; j < Ny; j++) {
            for (k = 0; k < Nz; k++) {
                in[(i*Ny + j)*Nz + k] = cos(i * L / Nx) * cos(j * L / Ny) * sin(k * L / Nz);
            }
        }
    }

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                in1[(i*ny + j)*nz + k] = in[(i*Ny + j)*Nz + k+1];
            }
        }
    }

    plan1 = fftw_plan_r2r_3d(nx, ny, nz, in1, out, FFTW_REDFT00, FFTW_REDFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(plan1);
    
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                index = (i * ny + j) * nz + k;
                out[index] /= Nx * Ny * Nz;
            }
        }
    }

    plan2 = fftw_plan_r2r_3d(nx, ny, nz, out, res1, FFTW_REDFT00, FFTW_REDFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(plan2);

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = nz; k > 0; k--) {
                res2[(i * Ny + j) * Nz + k] = res1[(i * ny + j) * nz + k-1];
            }
            res2[(i * Ny + j) * Nz] =  0;
            for(int k = 0; k < nz; k++){
                res2[(i * Ny + j) * Nz + Nz - k -1] = -res1[(i * ny + j) * nz + k];
            }
            res2[(i * Ny + j) * Nz + nz+1] =  0;
        }
    }

    for (i = 0; i < Nx; i++) {
        for (j = 1; j < ny; j++) {
            for (k = 0; k < Nz; k++) {
                res2[(i * Ny + Ny-j) * Nz + k] = res2[(i * Ny + j) * Nz + k];
            }
        }
    }

    for (i = 1; i < nx; i++) {
        for (j = 0; j < Ny; j++) {
            for (k = 0; k < Nz; k++) {
                res2[((Nx-i) * Ny + j) * Nz + k] = res2[(i * Ny + j) * Nz + k];
            }
        }
    }
    for (i = 0; i < Nx; i++) {
        for (j = 0; j < Ny; j++) {
            for (k = 0; k < Nz; k++) {
                //cout << i << " " << j << " " << k << " " <<in[(i*Ny + j)*Nz + k] << " " << res2[(i*Ny + j)*Nz + k] << endl;
            }
        }
    }
    
    double err = 0.0;
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                double err_ = fabs(res2[(i * ny + j) * nz + k] - in[(i * ny + j) * nz + k]);
                err += err_ * err_;
            }
        }  
    }
    err = sqrt(err);
    
    cout << "||u_out - u_in||_2 = " << err << endl;
    
    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);
    fftw_free(res1);
    fftw_free(res2);
    fftw_free(out);
    fftw_free(in1);
    fftw_free(in);
    return 0; 
}

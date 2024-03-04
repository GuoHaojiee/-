/*---------------------------------------------------------------------
    DST - IDST 3d шаг2: Cчитать производные  DST - IDST 3d
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
    int Nx = 32; 
    int Ny = 32; 
    int Nz = 32;
    int nx = (Nx/2+1);
    int ny = (Ny/2+1);
    int nz = (Nz/2-1);
    double L = 2 * M_PI;
    fftw_plan plan1, plan2, plan3;
    int i, j, k, index;

    double *in = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *x = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *x_1 = (double*) fftw_malloc(sizeof(double) * Nx * Ny * Nz);
    double *y = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *y_1 = (double*) fftw_malloc(sizeof(double) * Nx * Ny * Nz);
    double *z = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *z_1 = (double*) fftw_malloc(sizeof(double) * Nx * Ny * Nz);

    double * du2_dx2 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double * du2_dy2 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double * du2_dz2 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                in[(i*ny + j)*nz + k] = cos(i * L / Nx) * cos(j * L / Ny) * sin((k+1) * L / Nz);
            }
        }
    }

    plan1 = fftw_plan_r2r_3d(nx, ny, nz, in, z, FFTW_REDFT00, FFTW_REDFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(plan1);
    
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                index = (i * ny + j) * nz + k;
                z[index] /= Nx * Ny * Nz;
            }
        }
    }

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                index = (i * ny + j) * nz + k;
                x[index] = z[index] * i * i * (-1);
                y[index] = z[index] * j * j * (-1);
                z[index] = z[index] * (k+1) * (k+1) * (-1);
            }
        }
    }

    plan1 = fftw_plan_r2r_3d(nx, ny, nz, x, du2_dx2, FFTW_REDFT00, FFTW_REDFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    plan2 = fftw_plan_r2r_3d(nx, ny, nz, y, du2_dy2, FFTW_REDFT00, FFTW_REDFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    plan3 = fftw_plan_r2r_3d(nx, ny, nz, z, du2_dz2, FFTW_REDFT00, FFTW_REDFT00, FFTW_RODFT00, FFTW_ESTIMATE);

    fftw_execute(plan1);
    fftw_execute(plan2);
    fftw_execute(plan3);

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = nz; k > 0; k--) {
                x_1[(i * Ny + j) * Nz + k] = du2_dx2[(i * ny + j) * nz + k-1];
                y_1[(i * Ny + j) * Nz + k] = du2_dy2[(i * ny + j) * nz + k-1];
                z_1[(i * Ny + j) * Nz + k] = du2_dz2[(i * ny + j) * nz + k-1];
            }
            x_1[(i * Ny + j) * Nz] =  0;
            y_1[(i * Ny + j) * Nz] =  0;
            z_1[(i * Ny + j) * Nz] =  0;
            for(int k = 0; k < nz; k++){
                x_1[(i * Ny + j) * Nz + Nz - k -1] = -du2_dx2[(i * ny + j) * nz + k];
                y_1[(i * Ny + j) * Nz + Nz - k -1] = -du2_dy2[(i * ny + j) * nz + k];
                z_1[(i * Ny + j) * Nz + Nz - k -1] = -du2_dz2[(i * ny + j) * nz + k];
            }
            x_1[(i * Ny + j) * Nz + nz+1] =  0;
            y_1[(i * Ny + j) * Nz + nz+1] =  0;
            z_1[(i * Ny + j) * Nz + nz+1] =  0;
        }
    }

    for (i = 0; i < Nx; i++) {
        for (j = 1; j < ny; j++) {
            for (k = 0; k < Nz; k++) {
                x_1[(i * Ny + Ny-j) * Nz + k] = x_1[(i * Ny + j) * Nz + k];
                y_1[(i * Ny + Ny-j) * Nz + k] = y_1[(i * Ny + j) * Nz + k];
                z_1[(i * Ny + Ny-j) * Nz + k] = z_1[(i * Ny + j) * Nz + k];
            }
        }
    }

    for (i = 1; i < nx; i++) {
        for (j = 0; j < Ny; j++) {
            for (k = 0; k < Nz; k++) {
                x_1[((Nx-i) * Ny + j) * Nz + k] = x_1[(i * Ny + j) * Nz + k];
                y_1[((Nx-i) * Ny + j) * Nz + k] = y_1[(i * Ny + j) * Nz + k];
                z_1[((Nx-i) * Ny + j) * Nz + k] = z_1[(i * Ny + j) * Nz + k];
            }
        }
    }

    double err_x = 0.0, err_y = 0.0 , err_z = 0.0;
    for (i = 0; i < Nx; i++) {
        for (j = 0; j < Ny; j++) {
            for (k = 0; k < Nz; k++) {
                double du_dx2 = -1 * cos(i * L / Nx) * cos(j * L / Ny) * sin(k * L / Nz);
                double du_dy2 = -1 * cos(i * L / Nx) * cos(j * L / Ny) * sin(k * L / Nz);
                double du_dz2 = -1 * cos(i * L / Nx) * cos(j * L / Ny) * sin(k * L / Nz);
                //cout << y_1[(i * Ny + j) * Nz + k] << " " << du_dz2 << endl;
                double err_x_ = fabs(x_1[(i * Ny + j) * Nz + k] - du_dx2);
                double err_y_ = fabs(y_1[(i * Ny + j) * Nz + k] - du_dy2);
                double err_z_ = fabs(z_1[(i * Ny + j) * Nz + k] - du_dz2);
                err_x += err_x_ * err_x_;
                err_y += err_y_ * err_y_;
                err_z += err_z_ * err_z_;
            }
        }  
    }
    err_x = sqrt(err_x); err_y = sqrt(err_y); err_z = sqrt(err_z);
    cout << "||du/dx_out - du/dx||_2 = " << err_x << endl;
    cout << "||du/dy_out - du/dy||_2 = " << err_y << endl;
    cout << "||du/dz_out - du/dz||_2 = " << err_z << endl;
    
    fftw_destroy_plan(plan3);
    fftw_destroy_plan(plan2);
    fftw_destroy_plan(plan1);
    fftw_free(du2_dz2);
    fftw_free(du2_dy2);
    fftw_free(du2_dx2);
    fftw_free(z);
    fftw_free(y);
    fftw_free(x);
    return 0;
}
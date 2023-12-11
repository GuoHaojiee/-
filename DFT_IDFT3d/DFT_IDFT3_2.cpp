/*---------------------
    DFT - IDFT 3d шаг2: Cчитать производные 
  ---------------------  
    u(x,y,z) = sin4x * cos5y * sin3z
    du/dx = 4* cos4x * cos5y * sin3z 
    du/dy =-5* sin4x * sin5y * sin3z
    du/dz = 3* sin4x * cos5y * cos3z
    x, y ,z -(0, L) L = 2pi
*/ 
#include <iostream>
#include <fftw3.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>

using namespace std;

int main() {
    int nx = 64; 
    int ny = 64; 
    int nz = 64;
    double L = 2 * M_PI;
    fftw_plan plan1, plan2, plan3;
    int i, j, k, index;

    double *in_x = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *in_y = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *in_z = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *out_x = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *out_y = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *out_z = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    fftw_complex *in_complex_x = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *in_complex_y = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *in_complex_z = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                in_z[(i*ny + j)*nz + k] = sin(4.0 * i * L / nx) * cos(5.0 * j * L / nx) * sin(3.0 * k * L / nx);
            }
        }
    }

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                in_x[(i*ny + j)*nz + k] = in_z[(k*ny + j)*nz + i];
                in_y[(i*ny + j)*nz + k] = in_z[(i*ny + k)*nz + j];
            }
        }
    }

    plan1 = fftw_plan_dft_r2c_3d(nz, ny, nx, in_x, in_complex_x, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_r2c_3d(nx, nz, ny, in_y, in_complex_y, FFTW_ESTIMATE);
    plan3 = fftw_plan_dft_r2c_3d(nx, ny, nz, in_z, in_complex_z, FFTW_ESTIMATE);

    fftw_execute(plan1);
    fftw_execute(plan2);
    fftw_execute(plan3);

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < (nz / 2 + 1); k++) {
                index = (i * ny + j) * (nz / 2 + 1) + k;
                in_complex_x[index][0] /= (nx * ny * nz);
                in_complex_x[index][1] /= (nx * ny * nz);
                double real = in_complex_x[index][1] * k * (-1) ;
                double imaginary = in_complex_x[index][0] * k ;
                in_complex_x[index][0] = real;
                in_complex_x[index][1] = imaginary;
            }
        }
    }

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < (nz / 2 + 1); k++) {
                index = (i * ny + j) * (nz / 2 + 1) + k;
                in_complex_y[index][0] /= (nx * ny * nz);
                in_complex_y[index][1] /= (nx * ny * nz);
                double real = in_complex_y[index][1] * k * (-1) ;
                double imaginary = in_complex_y[index][0] * k ;
                in_complex_y[index][0] = real;
                in_complex_y[index][1] = imaginary;
            }
        }
    }

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < (nz / 2 + 1); k++) {
                index = (i * ny + j) * (nz / 2 + 1) + k;
                in_complex_z[index][0] /= (nx * ny * nz);
                in_complex_z[index][1] /= (nx * ny * nz);
                double real = in_complex_z[index][1] * k * k * (-1);
                double imaginary = in_complex_z[index][0] * k * k * (-1);
                in_complex_z[index][0] = imaginary;
                in_complex_z[index][1] = real;
            }
        }
    }

    plan1 = fftw_plan_dft_c2r_3d(nz, ny, nx, in_complex_x, out_x, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_c2r_3d(nx, nz, ny, in_complex_y, out_y, FFTW_ESTIMATE);
    plan3 = fftw_plan_dft_c2r_3d(nx, ny, nz, in_complex_z, out_z, FFTW_ESTIMATE);

    fftw_execute(plan1);
    fftw_execute(plan2);
    fftw_execute(plan3);

    double err_x = 0.0, err_y = 0.0 , err_z = 0.0;
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                double du_dx =  4 * cos(4.0 * k * L / nx) * cos(5.0 * j * L / ny) * sin(3.0 * i * L / nz);
                double du_dy =  -5 * sin(4.0 * i * L / nx) * sin(5.0 * k * L / ny) * sin(3.0 * j * L / nz);
                double du_dz = -9 * sin(4.0 * i * L / nx) * cos(5.0 * j * L / nx) * sin(3.0 * k * L / nx);
                //cout << out[(i * ny + j) * nz + k] << " " << du_dz << endl;
                double err_x_ = fabs(out_x[(i * ny + j) * nz + k] - du_dx);
                double err_y_ = fabs(out_y[(i * ny + j) * nz + k] - du_dy);
                double err_z_ = fabs(out_z[(i * ny + j) * nz + k] - du_dz);
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
    fftw_free(in_x);
    fftw_free(in_y);
    fftw_free(in_z);
    fftw_free(out_x);
    fftw_free(out_y);
    fftw_free(out_z);
    fftw_free(in_complex_x);
    fftw_free(in_complex_y);
    fftw_free(in_complex_z);
    return 0;
}


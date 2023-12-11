/*---------------------
    DFT - IDFT 3d шаг1: просто преобразование DFT - IDFT
  ---------------------  
    u(x,y,z) = sin4x * cos5y * sin3z
    x, y ,z -(0, L) L = 2pi
*/ 
#include <iostream>
#include <fftw3.h>
#include <math.h>
#include <stdlib.h>

using namespace std;

int main() {
    int nx = 64; 
    int ny = 64; 
    int nz = 64;
    double L = 2 * M_PI;
    fftw_plan plan;
    int i, j, k, index;

    double *in = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *out = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    fftw_complex *in_complex = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *out_complex = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                in[(i*ny + j)*nz + k] = sin(4.0 * i * L / nx) * cos(5.0 * j * L / ny) * sin(3.0 * k * L / nz);
            }
        }
    }

    plan = fftw_plan_dft_r2c_3d(nx, ny, nz, in, in_complex, FFTW_ESTIMATE);

    fftw_execute(plan);
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < (nz / 2 + 1); k++) {
                index = (i * ny + j) * (nz / 2 + 1) + k;
                in_complex[index][0] /= (nx * ny * nz);
                in_complex[index][1] /= (nx * ny * nz);
            }
        }
    }
    plan = fftw_plan_dft_c2r_3d(nx, ny, nz, in_complex, out, FFTW_ESTIMATE);

    fftw_execute(plan);

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                //cout << in[(i*ny + j)*nz + k] << " " << out[(i*ny + j)*nz + k] << endl;
            }
        }
    }

    double err = 0.0;
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                double err_ = fabs(out[(i * ny + j) * nz + k] - in[(i * ny + j) * nz + k]);
                err += err_ * err_;
            }
        }  
    }
    err = sqrt(err);
    
    cout << "||u_out - u_in||_2 = " << err << endl;

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    fftw_free(in_complex);
    fftw_free(out_complex);

    return 0;
}

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
    return (t + 1 ) * sin(4 * x) * cos(5 * y) * sin(3  * z);
}

double fi(double x, double y, double z) {
    return sin(4 * x) * cos(5 * y) * sin(3  * z);
}

double f_(double x, double y, double z, double t) {
    return (50 * t + 51) *sin(4 * x) * cos(5 * y) * sin(3  * z);
}

int main() {
    int nx = 32; 
    int ny = 32; 
    int nz = 32;
    int nt = 1000;
    double T = 1;
    double L = 2 * M_PI;

    double tau = T / nt;  

    fftw_plan plan1, plan2, plan3;
    int i, j, k, index;

    double *f = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *u = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *in = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    fftw_complex *f_complex = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *u_prev = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *u_next = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                in[(i*ny + j)*nz + k] = fi(i*L/nx, j*L/ny, k*L/nz);
            }
        }
    }

    plan1 = fftw_plan_dft_r2c_3d(nx, ny, nz, in, u_prev, FFTW_ESTIMATE);
    plan2 = fftw_plan_dft_r2c_3d(nx, ny, nz, f, f_complex, FFTW_ESTIMATE);
    fftw_execute(plan1);

    for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {  
                for (k = 0; k < (nz / 2 + 1); k++) {
                    index = (i * ny + j) * (nz / 2 + 1) + k;
                    u_prev[index][0] /= (nx * ny * nz);
                    u_prev[index][1] /= (nx * ny * nz);
                }
            }
        }
    
    for (int n = 1; n <= nt; n++) {
        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                for (k = 0; k < nz; k++) {
                    f[(i*ny + j)*nz + k] = f_(i*L/nx, j*L/ny, k*L/nz, (n-1)*T/nt);
                }
            }
        }
    
        fftw_execute(plan2);
    
        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {  
                for (k = 0; k < (nz / 2 + 1); k++) {
                    index = (i * ny + j) * (nz / 2 + 1) + k;
                    f_complex[index][0] /= (nx * ny * nz);
                    f_complex[index][1] /= (nx * ny * nz);
                }
            }
        }

        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                for (k = 0; k < (nz / 2 + 1); k++) {
                    index = (i * ny + j) * (nz / 2 + 1) + k;
                    double k_x = i <= nx/2 ? i : i -nx;
                    double k_y = j <= ny/2 ? j : j -ny;
                    double k_z = k;
                    u_next[index][0] = u_prev[index][0] + tau *((-1)* k_x * k_x * u_prev[index][0] + (-1)* k_y * k_y * u_prev[index][0] + (-1)* k_z * k_z * u_prev[index][0]) + tau * f_complex[index][0];
                    u_next[index][1] = u_prev[index][1] + tau *((-1)* k_x * k_x * u_prev[index][1] + (-1)* k_y * k_y * u_prev[index][1] + (-1)* k_z * k_z * u_prev[index][1]) + tau * f_complex[index][1];
                }
            }
        }

        for (i = 0; i < nx; i++) {
            for (j = 0; j < ny; j++) {
                for (k = 0; k < (nz / 2 + 1); k++) {
                    index = (i * ny + j) * (nz / 2 + 1) + k;
                    u_prev[index][0] = u_next[index][0];
                    u_prev[index][1] = u_next[index][1];
                }
            }
        }
    }

    plan3 = fftw_plan_dft_c2r_3d(nx, ny, nz, u_next, u, FFTW_ESTIMATE);
    fftw_execute(plan3);

    double err_x = 0.0, err_y = 0.0 , err_z = 0.0;
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                double res = (1 +  1) * sin(4.0 * i * L / nx) * cos(5.0 * j * L / ny) * sin(3.0 * k * L / nz);
                cout << u[(i * ny + j) * nz + k] << " " << res << endl;
                double err_x_ = fabs(u[(i * ny + j) * nz + k] - res);
                err_x += err_x_ * err_x_;
            }
        }  
    }
    err_x = sqrt(err_x);
    cout << "err = " << err_x << endl;
    
    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);
    fftw_destroy_plan(plan3);
    fftw_free(u_prev);
    fftw_free(u_next);
    fftw_free(f_complex);
    fftw_free(f);
    fftw_free(u);
    fftw_free(in);
    return 0;
}


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

double f_(double x, double y, double z, double t) {
    return (50*t*t+2*t+50) *sin(4 * x) * cos(5 * y) * sin(3  * z);
}


void euler(fftw_complex* u_prev, fftw_complex* u_next,  fftw_complex* f_1, 
                 double alpha,int N, double tau, double L, int nx, int ny, int nz) {
    double *f1 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    fftw_complex *u_prev_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_plan plan_f1 = fftw_plan_dft_r2c_3d(nx, ny, nz, f1, f_1, FFTW_ESTIMATE);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {  
            for (int k = 0; k < (nz / 2 + 1); k++) {
                int index = (i * ny + j) * (nz / 2 + 1) + k;
                u_prev_[index][0] = u_prev[index][0];
                u_prev_[index][1] = u_prev[index][1];
            }
        }
    }

    for (int n = 1; n <= N; n++)
    {
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    f1[(i*ny + j)*nz + k] = f_(i*L/nx, j*L/ny, k*L/nz, (n-1)*tau);
                }
            }
        }

        fftw_execute(plan_f1);

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < (nz / 2 + 1); k++) {
                    int index = (i * ny + j) * (nz / 2 + 1) + k;
                    int k_x = i <= nx/2 ? i : i -nx;
                    int k_y = j <= ny/2 ? j : j -ny;
                    int k_z = k;
                    u_next[index][0] = u_prev_[index][0] + tau *(-alpha * alpha *u_prev_[index][0] * (k_x * k_x + k_y * k_y + k_z * k_z) + f_1[index][0]);
                    u_next[index][1] = u_prev_[index][1] + tau *(-alpha * alpha *u_prev_[index][1] * (k_x * k_x + k_y * k_y + k_z * k_z) + f_1[index][1]);
                }
            }
        }

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < (nz / 2 + 1); k++) {
                    int index = (i * ny + j) * (nz / 2 + 1) + k;
                    u_prev_[index][0] = u_next[index][0];
                    u_prev_[index][1] = u_next[index][1];
                }
            }
        }
    }

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < (nz / 2 + 1); k++) {
                int index = (i * ny + j) * (nz / 2 + 1) + k;
                u_next[index][0] /= (nx * ny * nz);
                u_next[index][1] /= (nx * ny * nz);
            }
        }
    }
    fftw_destroy_plan(plan_f1);
    fftw_free(u_prev_);
    fftw_free(f1);
}

void calculateF(fftw_complex *f_1, fftw_complex *f_2, fftw_complex *f_3, double tau, double L, int n, int nx, int ny, int nz)
{
    double *f1 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *f2 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *f3 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    fftw_plan plan_f1 = fftw_plan_dft_r2c_3d(nx, ny, nz, f1, f_1, FFTW_ESTIMATE);
    fftw_plan plan_f2 = fftw_plan_dft_r2c_3d(nx, ny, nz, f2, f_2, FFTW_ESTIMATE);
    fftw_plan plan_f3 = fftw_plan_dft_r2c_3d(nx, ny, nz, f3, f_3, FFTW_ESTIMATE);

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
                int index = (i*ny + j)*nz + k;
                f1[index] = f_(i*L/nx, j*L/ny, k*L/nz, (n-1)*tau);
                f2[index] = f_(i*L/nx, j*L/ny, k*L/nz, (n-1) * tau + tau / 2);
                f3[index] = f_(i*L/nx, j*L/ny, k*L/nz, (n-1) * tau + tau);
            }
        }
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

void rungeKutta4(fftw_complex* u_prev, fftw_complex* u_next, fftw_complex* f_1, fftw_complex* f_2, fftw_complex* f_3, 
                 double alpha, int N, double tau, double L, int nx, int ny, int nz) {
    fftw_complex *k1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *k2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *k3 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *k4 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *u_prev_ = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {  
            for (int k = 0; k < (nz / 2 + 1); k++) {
                int index = (i * ny + j) * (nz / 2 + 1) + k;
                u_prev_[index][0] = u_prev[index][0];
                u_prev_[index][1] = u_prev[index][1];
            }
        }
    } 

    for (int n = 1; n <= N; n++) {
        calculateF(f_1,f_2,f_3,tau,L,n, nx,ny,nz);
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < (nz / 2 + 1); k++) {
                    int index = (i * ny + j) * (nz / 2 + 1) + k;
                    int k_x = i <= nx/2 ? i : i -nx;
                    int k_y = j <= ny/2 ? j : j -ny;
                    int k_z = k;
                    k1[index][0] = -alpha * alpha *u_prev_[index][0] * (k_x * k_x + k_y * k_y + k_z * k_z) + f_1[index][0];
                    k1[index][1] = -alpha * alpha *u_prev_[index][1] * (k_x * k_x + k_y * k_y + k_z * k_z) + f_1[index][1];
                    k2[index][0] = -alpha * alpha *(u_prev_[index][0] + tau / 2 * k1[index][0]) * (k_x * k_x + k_y * k_y + k_z * k_z) + f_2[index][0];
                    k2[index][1] = -alpha * alpha *(u_prev_[index][1] + tau / 2 * k1[index][1]) * (k_x * k_x + k_y * k_y + k_z * k_z) + f_2[index][1];
                    k3[index][0] = -alpha * alpha *(u_prev_[index][0] + tau / 2 * k2[index][0]) * (k_x * k_x + k_y * k_y + k_z * k_z) + f_2[index][0];
                    k3[index][1] = -alpha * alpha *(u_prev_[index][1] + tau / 2 * k2[index][1]) * (k_x * k_x + k_y * k_y + k_z * k_z) + f_2[index][1];
                    k4[index][0] = -alpha * alpha *(u_prev_[index][0] + tau * k3[index][0]) * (k_x * k_x + k_y * k_y + k_z * k_z) + f_3[index][0];
                    k4[index][1] = -alpha * alpha *(u_prev_[index][1] + tau * k3[index][1]) * (k_x * k_x + k_y * k_y + k_z * k_z) + f_3[index][1];
                }
            }
        }

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < (nz / 2 + 1); k++) {
                    int index = (i * ny + j) * (nz / 2 + 1) + k;
                    u_next[index][0] = u_prev_[index][0] + tau/6*(k1[index][0]+2*k2[index][0]+2*k3[index][0]+k4[index][0]);
                    u_next[index][1] = u_prev_[index][1] + tau/6*(k1[index][1]+2*k2[index][1]+2*k3[index][1]+k4[index][1]);
                }
            }
        }

        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < (nz / 2 + 1); k++) {
                    int index = (i * ny + j) * (nz / 2 + 1) + k;
                    u_prev_[index][0] = u_next[index][0];
                    u_prev_[index][1] = u_next[index][1];
                }
            }
        }
        
    }

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < (nz / 2 + 1); k++) {
                int index = (i * ny + j) * (nz / 2 + 1) + k;
                u_next[index][0] /= (nx * ny * nz);
                u_next[index][1] /= (nx * ny * nz);
            }
        }
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
    int nx = 16; 
    int ny = 16; 
    int nz = 16;
    int nt = 128;
    double T = 1;
    double L = 2 * M_PI;

    double tau = T / nt;  

    fftw_plan plan1, plan2;
    int i, j, k, index;

    double *in = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *out = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    double *res = (double*) fftw_malloc(sizeof(double) * nx * ny * nz);
    fftw_complex *f_1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *f_2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *f_3 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *u_prev = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));
    fftw_complex *u_next = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ny * (nz / 2 + 1));

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                in[(i*ny + j)*nz + k] = fi(i*L/nx, j*L/ny, k*L/nz);
                out[(i*ny + j)*nz + k] = u(i*L/nx, j*L/ny, k*L/nz, 1);
            }
        }
    }

    plan1 = fftw_plan_dft_r2c_3d(nx, ny, nz, in, u_prev, FFTW_ESTIMATE);
    fftw_execute(plan1);
    
    double alpha = 2 * M_PI / L;
    euler(u_prev,u_next, f_1,alpha, nt ,tau, L, nx, ny, nz);
    plan2 = fftw_plan_dft_c2r_3d(nx, ny, nz, u_next, res, FFTW_ESTIMATE);
    fftw_execute(plan2);
    double err_euler = err_calculation(out, res, nx*ny*nz);
    cout << "err_euler = " << err_euler << endl;
    
    rungeKutta4(u_prev,u_next, f_1, f_2, f_3,alpha,nt,tau,L,nx, ny, nz);
    fftw_execute(plan2);
    double err_rungeKutta4 = err_calculation(out, res, nx*ny*nz);
    cout << "err_rungeKutta4 = " << err_rungeKutta4 << endl;

    fftw_destroy_plan(plan1);
    fftw_destroy_plan(plan2);
    fftw_free(u_prev);
    fftw_free(u_next);
    fftw_free(f_1);
    fftw_free(f_2);
    fftw_free(f_3);
    fftw_free(in);
    fftw_free(res);
    fftw_free(out);
    return 0;
}


#include <iostream>
#include <fftw3.h>
#include <cmath>
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <omp.h>

using namespace std;

// Глобальные переменные
fftw_plan plan_fwd_r2r_cos;
fftw_plan plan_fwd_r2c_cos;
fftw_plan plan_bwd_c2r_cos;
fftw_plan plan_bwd_r2r_cos;

fftw_plan plan_fwd_r2r_sin;
fftw_plan plan_fwd_r2c_sin;
fftw_plan plan_bwd_c2r_sin;
fftw_plan plan_bwd_r2r_sin;

double *in;
double *in_z;
fftw_complex *complex_out;
double *re_out_xy;
double *re_out;

void initialize_fwd_r2r_cos(int nx, int ny, int nz1, int nz2) 
{   
    //Forward transformation для cos r2r
    int rank = 1;
    int n_cos[] = {nz1};
    int n_sin[] = {nz2};
    int howmany = nx * ny;
    int istride_cos = 1, ostride_cos = 1;
    int idist_cos = nz1, odist_cos = nz1;
    int *inembed_cos = n_cos, *onembed_cos = n_cos;
    const fftw_r2r_kind kind_cos[] = {FFTW_REDFT00};
    plan_fwd_r2r_cos = fftw_plan_many_r2r(rank, n_cos, howmany,
                            in, inembed_cos, istride_cos, idist_cos,
                            in_z, onembed_cos, ostride_cos, odist_cos,
                            kind_cos, FFTW_ESTIMATE);
}

void initialize_fwd_r2r_sin(int nx, int ny, int nz1, int nz2) 
{   
    //Forward transformation для sin r2r
    int rank = 1;
    int n_sin[] = {nz2};
    int howmany = nx * ny;
    int istride_sin = 1, ostride_sin = 1;
    int idist_sin = nz2, odist_sin = nz2;
    int *inembed_sin = n_sin, *onembed_sin = n_sin;
    const fftw_r2r_kind kind_sin[] = {FFTW_RODFT00};

    plan_fwd_r2r_sin = fftw_plan_many_r2r(rank, n_sin, howmany,
                            in, inembed_sin, istride_sin, idist_sin,
                            in_z, onembed_sin, ostride_sin, odist_sin,
                            kind_sin, FFTW_ESTIMATE);
}

void initialize_fwd_r2c_cos(int nx, int ny, int nz1, int nz2) 
{ 
    int nn[] = {nx, ny};
    int inembed2[] =  {nx, ny};
    int onembed2[] =  {nx, ny/2+1};
    int istride_cos = nz1, ostride_cos = nz1;
    int idist_cos = 1, odist_cos = 1;

    plan_fwd_r2c_cos = fftw_plan_many_dft_r2c(2, nn, nz1,
                                in_z, inembed2, istride_cos, idist_cos,
                                complex_out, onembed2,  ostride_cos, odist_cos,
                                FFTW_ESTIMATE);
}

void initialize_fwd_r2c_sin(int nx, int ny, int nz1, int nz2) 
{ 
    int nn[] = {nx, ny};
    int inembed2[] =  {nx, ny};
    int onembed2[] =  {nx, ny/2+1};
    int istride_sin = nz2, ostride_sin = nz2;
    int idist_sin = 1, odist_sin = 1;

    plan_fwd_r2c_sin = fftw_plan_many_dft_r2c(2, nn, nz2,
                                in_z, inembed2, istride_sin, idist_sin,
                                complex_out, onembed2, ostride_sin, odist_sin,
                                FFTW_ESTIMATE); 
}

void initialize_bwd_c2r_cos(int nx, int ny, int nz1, int nz2) 
{ 
    //Backward transformation для cos c2r
    int nn[] = {nx, ny};
    int inembed4[] =  {nx, (ny/2+1)};
    int onembed4[] =  {nx, ny};
    int istride_cos = nz1, ostride_cos = nz1;
    int idist_cos = 1, odist_cos = 1;

    plan_bwd_c2r_cos = fftw_plan_many_dft_c2r(2, nn, nz1,
                                complex_out, inembed4,istride_cos, idist_cos,
                                re_out_xy, onembed4,  ostride_cos, odist_cos,
                                FFTW_ESTIMATE);
}

void initialize_bwd_c2r_sin(int nx, int ny, int nz1, int nz2) 
{ 
    //Backward transformation для sin c2r
    int nn[] = {nx, ny};
    int inembed4[] =  {nx, (ny/2+1)};
    int onembed4[] =  {nx, ny};
    int istride_sin = nz2, ostride_sin = nz2;
    int idist_sin = 1, odist_sin = 1;
    plan_bwd_c2r_sin = fftw_plan_many_dft_c2r(2, nn, nz2,
                                complex_out, inembed4, istride_sin, idist_sin,
                                re_out_xy, onembed4, ostride_sin, odist_sin,
                                FFTW_ESTIMATE);
}

void initialize_bwd_r2r_cos(int nx, int ny, int nz1, int nz2) 
{ 
    int rank = 1;
    int n_cos[] = {nz1};
    int howmany = nx * ny;
    int istride_cos = 1, ostride_cos = 1;
    int idist_cos = nz1, odist_cos = nz1;
    int *inembed3_cos = n_cos, *onembed3_cos = n_cos;
    const fftw_r2r_kind kind_cos[] = {FFTW_REDFT00};

    plan_bwd_r2r_cos = fftw_plan_many_r2r(rank, n_cos, howmany,
                              re_out_xy, inembed3_cos,istride_cos, idist_cos,
                              re_out, onembed3_cos,  ostride_cos, odist_cos,
                              kind_cos, FFTW_ESTIMATE);
}

void initialize_bwd_r2r_sin(int nx, int ny, int nz1, int nz2) 
{ 
    int rank = 1;
    int n_sin[] = {nz2};
    int howmany = nx * ny;
    int istride_sin = 1, ostride_sin = 1;
    int idist_sin = nz2, odist_sin = nz2;
    int *inembed3_sin = n_sin, *onembed3_sin = n_sin;
    const fftw_r2r_kind kind_sin[] = {FFTW_RODFT00};

    plan_bwd_r2r_sin = fftw_plan_many_r2r(rank, n_sin, howmany,
                              re_out_xy, inembed3_sin, istride_sin, idist_sin,
                              re_out, onembed3_sin, ostride_sin, odist_sin,
                              kind_sin, FFTW_ESTIMATE);
}

void finalize_fft_plans() {
    fftw_destroy_plan(plan_fwd_r2r_cos);
    fftw_destroy_plan(plan_fwd_r2c_cos);
    fftw_destroy_plan(plan_bwd_c2r_cos);
    fftw_destroy_plan(plan_bwd_r2r_cos);

    fftw_destroy_plan(plan_fwd_r2r_sin);
    fftw_destroy_plan(plan_fwd_r2c_sin);
    fftw_destroy_plan(plan_bwd_c2r_sin);
    fftw_destroy_plan(plan_bwd_r2r_sin);
}

double func_V1(double x, double y, double z, double t) {
    return (t*t+1)*exp(sin(3*x+3*y))*cos(6*z);
}
double func_V2(double x, double y, double z, double t) {
    return (t*t+1)*exp(sin(3*x+3*y))*cos(6*z);
}
double func_V3(double x, double y, double z, double t) {
    return -(t*t+1)*exp(sin(3*x+3*y))*cos(3*x+3*y)*sin(6*z);
}

double func_dV1_dt(double x, double y, double z, double t) {
    return 2*t*exp(sin(3*x+3*y))*cos(6*z);
}

double func_dV2_dt(double x, double y, double z, double t) {
    return 2*t*exp(sin(3*x+3*y))*cos(6*z);
}

double func_dV3_dt(double x, double y, double z, double t) {
    return -2*t*exp(sin(3*x+3*y))*cos(3*x+3*y)*sin(6*z);
}

double func_laplace_V1(double x, double y, double z, double t) {
    double d2v1_dx2 = (t*t+1)*9*exp(sin(3*x+3*y))*((cos(3*x+3*y)*cos(3*x+3*y))-sin(3*x+3*y))*cos(6*z);
    double d2v1_dy2 = (t*t+1)*9*exp(sin(3*x+3*y))*((cos(3*x+3*y)*cos(3*x+3*y))-sin(3*x+3*y))*cos(6*z);
    double d2v1_dz2 = -(t*t+1)*36*exp(sin(3*x+3*y))*cos(6*z);
    return d2v1_dx2 + d2v1_dy2 + d2v1_dz2;
}

double func_laplace_V2(double x, double y, double z, double t) {
    return func_laplace_V1(x,y,z,t);
}

double func_laplace_V3(double x, double y, double z, double t) {
    double d2v3_dx2 = -(t*t+1)*9*exp(sin(3*x+3*y))*cos(3*x+3*y)*((cos(3*x+3*y)*cos(3*x+3*y)-sin(3*x+3*y))-(2*sin(3*x+3*y)+1))*sin(6*z);
    double d2v3_dy2 = -(t*t+1)*9*exp(sin(3*x+3*y))*cos(3*x+3*y)*((cos(3*x+3*y)*cos(3*x+3*y)-sin(3*x+3*y))-(2*sin(3*x+3*y)+1))*sin(6*z);
    double d2v3_dz2 = (t*t+1)*36*exp(sin(3*x+3*y))*cos(3*x+3*y)*sin(6*z);
    return d2v3_dx2 + d2v3_dy2 + d2v3_dz2;
}

double func_rot1(double x, double y, double z, double t){
    double dv3_dy = -(t*t+1)*3*exp(sin(3*x+3*y))*(cos(3*x+3*y)*cos(3*x+3*y)-sin(3*x+3*y))*sin(6*z);
    double dv2_dz = -(t*t+1)*6*exp(sin(3*x+3*y))*sin(6*z);
    return dv3_dy - dv2_dz;
}

double func_rot2(double x, double y, double z, double t){
    double dv1_dz = -(t*t+1)*6*exp(sin(3*x+3*y))*sin(6*z);
    double dv3_dx = -(t*t+1)*3*exp(sin(3*x+3*y))*(cos(3*x+3*y)*cos(3*x+3*y)-sin(3*x+3*y))*sin(6*z);
    return dv1_dz - dv3_dx;
}

double func_rot3(double x, double y, double z, double t){
    return 0;
}

double func_v_cross_rot1(double x, double y, double z, double t) {
    return func_V2(x,y,z,t)*func_rot3(x,y,z,t)-func_V3(x,y,z,t)*func_rot2(x,y,z,t);
}

double func_v_cross_rot2(double x, double y, double z, double t) {
    return func_V3(x,y,z,t)*func_rot1(x,y,z,t)-func_V1(x,y,z,t)*func_rot3(x,y,z,t);
}

double func_v_cross_rot3(double x, double y, double z, double t) {
    return func_V1(x,y,z,t)*func_rot2(x,y,z,t)-func_V2(x,y,z,t)*func_rot1(x,y,z,t);
}

double func_p(double x, double y, double z, double t) {
    return (t*t+1)*cos(x)*cos(y)*cos(z);
}

double func_grad_p1(double x, double y, double z, double t) {
    return -(t*t+1)*sin(x)*cos(y)*cos(z);
}
double func_grad_p2(double x, double y, double z, double t) {
    return -(t*t+1)*cos(x)*sin(y)*cos(z);
}
double func_grad_p3(double x, double y, double z, double t) {
    return -(t*t+1)*cos(x)*cos(y)*sin(z);
}

double func_f1(double x, double y, double z, double t) {
    return func_dV1_dt(x,y,z,t)- func_laplace_V1(x,y,z,t)- func_v_cross_rot1(x,y,z,t) + func_grad_p1(x,y,z,t); 
}

double func_f2(double x, double y, double z, double t) {
    return func_dV2_dt(x,y,z,t)- func_laplace_V2(x,y,z,t)- func_v_cross_rot2(x,y,z,t) + func_grad_p2(x,y,z,t); 
}

double func_f3(double x, double y, double z, double t) {
    return func_dV3_dt(x,y,z,t)- func_laplace_V3(x,y,z,t)- func_v_cross_rot3(x,y,z,t) + func_grad_p3(x,y,z,t); 
}

void normalization(fftw_complex* V_c_, int nx, int ny, int nz, int factor)
{   
    int i, j, k, index;
    #pragma omp parallel for
    for(int i = 0; i < nx; ++i) {
        for(int j = 0; j < ny; ++j) {
            for(int k = 0; k < nz; ++k) {
                int index = (i*ny+j)*nz+k;
                //Normalization
                V_c_[index][0] /= factor;
                V_c_[index][1] /= factor;
            }
        }
    }
}

void compute_rot(fftw_complex* V1_c_, fftw_complex* V2_c_, fftw_complex* V3_c_
                ,fftw_complex* rot1_c_, fftw_complex* rot2_c_, fftw_complex* rot3_c_, int nx, int ny, int nz, double L_x, double L_y, double L_z)
{
    int i, j, k, index,k_x,k_y,k_z,index1,index2;
    double alpha = 1;
    #pragma omp parallel for
    for(int i = 0; i < nx; ++i) {
        for(int j = 0; j < (ny/2+1); ++j) {
            for(int k = 0; k < (nz/2+1); ++k) {
                int index = (i * (ny/2+1) + j) * (nz/2+1) + k;
                int k_x = i <= nx/2 ? i : i -nx;
                int k_y = j <= ny/2 ? j : j -ny;
                int k_z = k;

                rot3_c_[index][0] = -(V2_c_[index][1] * k_x - V1_c_[index][1] * k_y); 
                rot3_c_[index][1] = V2_c_[index][0] * k_x - V1_c_[index][0] * k_y;      
            }
        }
    }
    
    #pragma omp parallel for
    for(int i = 0; i < nx; ++i) {
        for(int j = 0; j < (ny/2+1); ++j) {
            for(int k = 0; k < (nz/2-1); ++k) {
                int index1 = (i * (ny/2+1) + j) * (nz/2+1) + k + 1;
                int index2 = (i * (ny/2+1) + j) * (nz/2-1) + k;

                int k_x = i <= nx/2 ? i : i -nx;
                int k_y = j <= ny/2 ? j : j -ny;
                int k_z = k;

                rot1_c_[index2][0] = (-V3_c_[index2][1] * (k_y)) -(-V2_c_[index1][0] * (k_z+1)*alpha); 
                rot1_c_[index2][1] = (V3_c_[index2][0] * (k_y)) - (-V2_c_[index1][1] * (k_z+1) *alpha); 
                rot2_c_[index2][0] = (-V1_c_[index1][0] * (k_z+1)*alpha) - (-V3_c_[index2][1] * (k_x)); 
                rot2_c_[index2][1] = (-V1_c_[index1][1] * (k_z+1)*alpha) - (V3_c_[index2][0] * (k_x));  
            }
        }
    }
}

void compute_div(fftw_complex* V1_c_, fftw_complex* V2_c_, fftw_complex* V3_c_
                ,fftw_complex* div_c_, int nx, int ny, int nz1, int nz2 ,double L_x, double L_y, double L_z)
{
    int i,j,k,index,k_x,k_y,k_z,index1,index2;
    double alpha = 2*M_PI/L_z;
    #pragma omp parallel for
    for(int i = 0; i < nx; ++i) {
        for(int j = 0; j < (ny/2+1); ++j) {
            int k_x = i <= nx/2 ? i : i -nx;
            int k_y = j <= ny/2 ? j : j -ny;
            for(int k = 1; k <= nz2; ++k) {
                int index1 = (i * (ny/2+1) + j) * nz1 + k;
                int index2 = (i * (ny/2+1) + j) * nz2 + k-1;
                int k_z = k;
                // Divergence:
                div_c_[index1][0] = (-V1_c_[index1][1] * k_x) + (-V2_c_[index1][1] * k_y) + (V3_c_[index2][0] * k_z*alpha);
                div_c_[index1][1] = (V1_c_[index1][0] * k_x) +  (V2_c_[index1][0] * k_y ) + (V3_c_[index2][1] * k_z*alpha); 
            }
            // k = 0
            int index = (i * (ny/2+1) + j) * nz1 + 0;
            div_c_[index][0] = (-V1_c_[index][1] * k_x) + (-V2_c_[index][1] * k_y);
            div_c_[index][1] = (V1_c_[index][0] * k_x) +  (V2_c_[index][0] * k_y );

            // k = nz/2
            int index2 = (i * (ny/2+1) + j) * nz1 + nz1-1;
            div_c_[index2][0] = (-V1_c_[index2][1] * k_x) + (-V2_c_[index2][1] * k_y);
            div_c_[index2][1] = (V1_c_[index2][0] * k_x) +  (V2_c_[index2][0] * k_y );
        }
    }
}

void make_div_0(fftw_complex* V1_, fftw_complex* V2_, fftw_complex* V3_, fftw_complex* fi, fftw_complex* div_,
                int nx, int ny,int nz1, int nz2,double L_x, double L_y, double L_z)
{
    int i, j, k,k_x,k_y,k_z,index1,index2;
    double alpha = 2*M_PI/L_z;
    compute_div(V1_,V2_,V3_,div_,nx,ny,nz1,nz2,L_x,L_y,L_z);

    #pragma omp parallel for
    for(int i = 0; i < nx; ++i) {
        for(int j = 0; j < (ny/2+1); ++j) {
            for(int k = 0; k < nz1; ++k) {
                int k_x = i <= nx/2 ? i : i -nx;
                int k_y = j <= ny/2 ? j : j -ny;
                int k_z = k;
                int index1 = (i * (ny/2+1) + j) * nz1 + k;

                if (i==0 && j==0 && k==0) {
                    fi[index1][0] = 0;
                    fi[index1][1] = 0;
                } else {
                    fi[index1][0] = div_[index1][0]/ -(k_x*k_x + k_y*k_y + k_z*k_z*alpha*alpha);
                    fi[index1][1] = div_[index1][1]/ -(k_x*k_x + k_y*k_y + k_z*k_z*alpha*alpha);
                }
            }
        }
    }

    #pragma omp parallel for
    for(int i = 0; i < nx; ++i) {
        for(int j = 0; j < (ny/2+1); ++j) {
            for(int k = 0; k < nz1; ++k) {
                int k_x = i <= nx/2 ? i : i -nx;
                int k_y = j <= ny/2 ? j : j -ny;
                int k_z = k;
                int index1 = (i * (ny/2+1) + j) * nz1 + k;
                int index2 = (i * (ny/2+1) + j) * nz2 + k-1;

                if (k != 0 && k != nz1 - 1) 
                {
                    V1_[index1][0] = V1_[index1][0] - (-fi[index1][1] * k_x);
                    V1_[index1][1] = V1_[index1][1] - (fi[index1][0] * k_x);
                    V2_[index1][0] = V2_[index1][0] - (-fi[index1][1] * k_y);
                    V2_[index1][1] = V2_[index1][1] - (fi[index1][0] * k_y);
                    V3_[index2][0] = V3_[index2][0] - (-fi[index1][0]* k_z*alpha);
                    V3_[index2][1] = V3_[index2][1] - (-fi[index1][1]* k_z*alpha);
                } 
                else if (k == 0 || k == nz1 - 1) 
                {
                    V1_[index1][0] = V1_[index1][0] - (-fi[index1][1] * k_x);
                    V1_[index1][1] = V1_[index1][1] - (fi[index1][0] * k_x);
                    V2_[index1][0] = V2_[index1][0] - (-fi[index1][1] * k_y);
                    V2_[index1][1] = V2_[index1][1] - (fi[index1][0] * k_y);
                }
            }
        }
    }
}



void compute_v_cross_rot(fftw_complex* V1_c, fftw_complex *V2_c, fftw_complex *V3_c
                        ,double *V1_r, double *V2_r, double *V3_r
                        ,double *V1_xy_r, double *V2_xy_r, double *V3_xy_r
                        ,fftw_complex *rotv1_c, fftw_complex *rotv2_c, fftw_complex *rotv3_c
                        ,double *rotv1_r, double *rotv2_r, double *rotv3_r
                        ,double *rotv1_xy_r, double *rotv2_xy_r, double *rotv3_xy_r
                        ,fftw_complex *v_cross_rot1_c, fftw_complex *v_cross_rot2_c, fftw_complex *v_cross_rot3_c
                        ,double *v_cross_rot1_r, double *v_cross_rot2_r, double *v_cross_rot3_r
                        ,double *v_cross_rot1_z_r, double *v_cross_rot2_z_r, double *v_cross_rot3_z_r
                        ,int nx, int ny, int nz1, int nz2,int Nz,int L_x, int L_y, int L_z)
{   
    int index1,index2;
    compute_rot(V1_c, V2_c, V3_c, rotv1_c, rotv2_c, rotv3_c, nx, ny, Nz,L_x,L_y,L_z);

    // backward fft for rotV
    fftw_execute_dft_c2r(plan_bwd_c2r_sin,rotv1_c,rotv1_xy_r);
    fftw_execute_r2r(plan_bwd_r2r_sin, rotv1_xy_r, rotv1_r);

    fftw_execute_dft_c2r(plan_bwd_c2r_sin,rotv2_c,rotv2_xy_r);
    fftw_execute_r2r(plan_bwd_r2r_sin, rotv2_xy_r, rotv2_r);
    
    fftw_execute_dft_c2r(plan_bwd_c2r_cos,rotv3_c,rotv3_xy_r);
    fftw_execute_r2r(plan_bwd_r2r_cos, rotv3_xy_r, rotv3_r);

    #pragma omp parallel for
    for (int i = 0; i < nx; ++i){
        for (int j = 0; j < (ny / 2 + 1); ++j){
            for (int k = 0; k < nz1; ++k){
                int index = (i * (ny / 2 + 1) + j) * nz1 + k;
                v_cross_rot1_c[index][0] = V1_c[index][0];
                v_cross_rot1_c[index][1] = V1_c[index][1];
                v_cross_rot2_c[index][0] = V2_c[index][0];
                v_cross_rot2_c[index][1] = V2_c[index][1];
            }
            for (int k = 0; k < nz2; ++k){
                int index = (i * (ny / 2 + 1) + j) * nz2 + k;
                v_cross_rot3_c[index][0] = V3_c[index][0];
                v_cross_rot3_c[index][1] = V3_c[index][1];
            }
        }
    }

    // backward fft for V
    fftw_execute_dft_c2r(plan_bwd_c2r_cos,v_cross_rot1_c,V1_xy_r);
    fftw_execute_r2r(plan_bwd_r2r_cos, V1_xy_r, V1_r);

    fftw_execute_dft_c2r(plan_bwd_c2r_cos,v_cross_rot2_c,V2_xy_r);
    fftw_execute_r2r(plan_bwd_r2r_cos, V2_xy_r, V2_r);
    
    fftw_execute_dft_c2r(plan_bwd_c2r_sin,v_cross_rot3_c,V3_xy_r);
    fftw_execute_r2r(plan_bwd_r2r_sin, V3_xy_r, V3_r);

    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz1; k++) {
                int index1 = (i*ny+j)*nz1+k;
                int index2 = (i*ny+j)*nz2+k-1;
                if (k != 0 && k != nz1 - 1) 
                {
                    
                    v_cross_rot1_r[index1] = V2_r[index1]*rotv3_r[index1] - V3_r[index2]*rotv2_r[index2];
                    v_cross_rot2_r[index1] = V3_r[index2]*rotv1_r[index2] - V1_r[index1]*rotv3_r[index1];
                    v_cross_rot3_r[index2] = V1_r[index1]*rotv2_r[index2] - V2_r[index1]*rotv1_r[index2];
                } else 
                {
                    v_cross_rot1_r[index1] = V2_r[index1]*rotv3_r[index1] - 0;
                    v_cross_rot2_r[index1] = 0 - V1_r[index1]*rotv3_r[index1];
                }
            }      
        }
    }
    // Forward transformation  
    // Сделал fft для V x rotV, 
    fftw_execute_r2r(plan_fwd_r2r_cos, v_cross_rot1_r, v_cross_rot1_z_r);
    fftw_execute_dft_r2c(plan_fwd_r2c_cos, v_cross_rot1_z_r, v_cross_rot1_c);

    fftw_execute_r2r(plan_fwd_r2r_cos, v_cross_rot2_r, v_cross_rot2_z_r);
    fftw_execute_dft_r2c(plan_fwd_r2c_cos, v_cross_rot2_z_r, v_cross_rot2_c);

    fftw_execute_r2r(plan_fwd_r2r_sin, v_cross_rot3_r, v_cross_rot3_z_r);
    fftw_execute_dft_r2c(plan_fwd_r2c_sin, v_cross_rot3_z_r, v_cross_rot3_c);

    normalization(v_cross_rot1_c, nx,(ny/2+1), nz1, nx*ny*Nz);
    normalization(v_cross_rot2_c, nx,(ny/2+1), nz1, nx*ny*Nz);
    normalization(v_cross_rot3_c, nx,(ny/2+1), nz2, nx*ny*Nz);
}

void compute_f(fftw_complex* f1_c_, fftw_complex* f2_c_, fftw_complex* f3_c_,double* f1_z_r_,double* f2_z_r_,double* f3_z_r_,
            double* f1_r_,double* f2_r_,double* f3_r_, int nx, int ny, int nz1,int nz2,int Nz,double L_x, double L_y, double L_z,int nt, double tau, int it, double c)
{   
    int i, j, k, index,k_x,k_y,k_z,index1,index2;
    #pragma omp parallel for
    for (int i=0; i<nx; i++){
        for (int j=0; j<ny; j++){
            for (int k = 0; k < nz1; k++){
                int index1 = (i * ny + j) * nz1 + k;
                f1_r_[index1] = func_f1(i * L_x / nx, j * L_y / ny, k * L_z / Nz, it*tau + c);
                f2_r_[index1] = func_f2(i * L_x / nx, j * L_y / ny, k * L_z / Nz, it*tau + c);
            }   
            for (int k = 1; k <= nz2; k++) {
                int index2 = (i*ny+j)*nz2+k-1;
                f3_r_[index2] = func_f3(i*L_x/nx, j*L_y/ny, k*L_z/Nz, it*tau + c);
            } 
        }
    }
    // f
    fftw_execute_r2r(plan_fwd_r2r_cos, f1_r_, f1_z_r_);
    fftw_execute_dft_r2c(plan_fwd_r2c_cos, f1_z_r_, f1_c_);

    fftw_execute_r2r(plan_fwd_r2r_cos, f2_r_, f2_z_r_);
    fftw_execute_dft_r2c(plan_fwd_r2c_cos, f2_z_r_, f2_c_);

    fftw_execute_r2r(plan_fwd_r2r_sin, f3_r_, f3_z_r_);
    fftw_execute_dft_r2c(plan_fwd_r2c_sin, f3_z_r_, f3_c_);

    normalization(f1_c_, nx, (ny / 2 + 1), nz1, nx * ny * Nz);
    normalization(f2_c_, nx, (ny / 2 + 1), nz1, nx * ny * Nz);
    normalization(f3_c_, nx, (ny / 2 + 1), nz2, nx * ny * Nz);
}

void compute_F(fftw_complex* F1_c_, fftw_complex* F2_c_, fftw_complex* F3_c_,
            fftw_complex* f1_c_,fftw_complex* f2_c_,fftw_complex* f3_c_,
            double* f1_z_r_,double* f2_z_r_,double* f3_z_r_,
            double* f1_r_,double* f2_r_,double* f3_r_,
            double *V1_xy_r, double *V2_xy_r, double *V3_xy_r,
            fftw_complex *rot1_c, fftw_complex *rot2_c, fftw_complex *rot3_c,
            double *rot1_r, double *rot2_r, double *rot3_r,
            double *rot1_xy_r, double *rot2_xy_r, double *rot3_xy_r,
            fftw_complex* cross1_c_, fftw_complex* cross2_c_, fftw_complex* cross3_c_,
            double *cross1_r, double *cross2_r, double *cross3_r,
            double *cross1_z_r, double *cross2_z_r, double *cross3_z_r,
            fftw_complex* p_c,fftw_complex* div_c,
            int nx, int ny, int nz1,int nz2,int Nz,double L_x, double L_y, double L_z,int nt, double tau, int it, double c)
{   
    compute_v_cross_rot(F1_c_,F2_c_,F3_c_,f1_r_,f2_r_,f3_r_,V1_xy_r,V2_xy_r,V3_xy_r,rot1_c,rot2_c,rot3_c,rot1_r,rot2_r,rot3_r,rot1_xy_r,rot2_xy_r,rot3_xy_r,
                        cross1_c_,cross2_c_,cross3_c_,cross1_r,cross2_r,cross3_r,cross1_z_r,cross2_z_r,cross3_z_r,nx,ny,nz1,nz2,Nz,L_x,L_y,L_z);
    compute_f(f1_c_,f2_c_,f3_c_,f1_z_r_,f2_z_r_,f3_z_r_,f1_r_,f2_r_,f3_r_,nx,ny,nz1,nz2,Nz,L_x,L_y,L_z,nt,tau,it,c);

    #pragma omp parallel for
    for(int i = 0; i < nx; ++i) {
        for(int j = 0; j < (ny/2+1); ++j) {
            for(int k = 0; k < nz1; ++k){   
                int k_x = i <= nx/2 ? i : i -nx;
                int k_y = j <= ny / 2 ? j : j - ny;
                int k_z = k;
                int index1= (i*(ny/2+1)+j)*nz1+k;
                F1_c_[index1][0] = -F1_c_[index1][0] * (k_x * k_x + k_y * k_y + k_z * k_z) + cross1_c_[index1][0] + f1_c_[index1][0];
                F1_c_[index1][1] = -F1_c_[index1][1] * (k_x * k_x + k_y * k_y + k_z * k_z) + cross1_c_[index1][1] + f1_c_[index1][1];
                F2_c_[index1][0] = -F2_c_[index1][0] * (k_x * k_x + k_y * k_y + k_z * k_z) + cross2_c_[index1][0] + f2_c_[index1][0];
                F2_c_[index1][1] = -F2_c_[index1][1] * (k_x * k_x + k_y * k_y + k_z * k_z) + cross2_c_[index1][1] + f2_c_[index1][1];
            }   
            for (int k = 0; k < nz2; ++k){
                int k_x = i <= nx/2 ? i : i -nx;
                int k_y = j <= ny / 2 ? j : j - ny;
                int k_z = k;
                int index2 = (i*(ny/2+1)+j)*nz2+k;
                F3_c_[index2][0] = -F3_c_[index2][0] * (k_x * k_x + k_y * k_y + (k_z+1) * (k_z+1)) + cross3_c_[index2][0] + f3_c_[index2][0];
                F3_c_[index2][1] = -F3_c_[index2][1] * (k_x * k_x + k_y * k_y + (k_z+1) * (k_z+1)) + cross3_c_[index2][1] + f3_c_[index2][1];
            } 
        }
    }
    make_div_0(F1_c_, F2_c_, F3_c_, p_c,div_c,nx, ny, nz1, nz2, L_x, L_y, L_z);
}

void rungeKutta(fftw_complex* V1_c_, fftw_complex* V2_c_, fftw_complex* V3_c_,
            fftw_complex* F1_c_, fftw_complex* F2_c_, fftw_complex* F3_c_,
            fftw_complex* f1_c_,fftw_complex* f2_c_,fftw_complex* f3_c_,
            double* f1_z_r_,double* f2_z_r_,double* f3_z_r_,
            double* f1_r_,double* f2_r_,double* f3_r_,
            double *V1_xy_r, double *V2_xy_r, double *V3_xy_r,
            fftw_complex *rot1_c, fftw_complex *rot2_c, fftw_complex *rot3_c,
            double *rot1_r, double *rot2_r, double *rot3_r,
            double *rot1_xy_r, double *rot2_xy_r, double *rot3_xy_r,
            fftw_complex* cross1_c_, fftw_complex* cross2_c_, fftw_complex* cross3_c_,
            double *cross1_r, double *cross2_r, double *cross3_r,
            double *cross1_z_r, double *cross2_z_r, double *cross3_z_r,
            fftw_complex* p_c,fftw_complex* div_c,
            int nx, int ny, int nz1,int nz2,int Nz,double L_x, double L_y, double L_z,int nt, double tau)
{
    fftw_complex *k1_V1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *k2_V1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *k3_V1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *k4_V1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);

    fftw_complex *k1_V2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *k2_V2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *k3_V2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *k4_V2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);

    fftw_complex *k1_V3 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *k2_V3 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *k3_V3 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *k4_V3 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);

    for (int l = 0; l < nt; l++) {
        #pragma omp parallel for
        for (int i = 0; i < nx; ++i){
            for (int j = 0; j < (ny / 2 + 1); ++j){
                for (int k = 0; k < nz1; ++k){
                    int index = (i * (ny / 2 + 1) + j) * nz1 + k;
                    F1_c_[index][0] = V1_c_[index][0];
                    F1_c_[index][1] = V1_c_[index][1];
                    F2_c_[index][0] = V2_c_[index][0];
                    F2_c_[index][1] = V2_c_[index][1];
                }
                for (int k = 0; k < nz2; ++k){
                    int index = (i * (ny / 2 + 1) + j) * nz2 + k;
                    F3_c_[index][0] = V3_c_[index][0];
                    F3_c_[index][1] = V3_c_[index][1];
                }
            }
        }
        compute_F(F1_c_,F2_c_,F3_c_,f1_c_,f2_c_,f3_c_,f1_z_r_,f2_z_r_,f3_z_r_,f1_r_,f2_r_,f3_r_,V1_xy_r,V2_xy_r,V3_xy_r,rot1_c,rot2_c,rot3_c,rot1_r,rot2_r,rot3_r,rot1_xy_r,rot2_xy_r,rot3_xy_r,
                            cross1_c_,cross2_c_,cross3_c_,cross1_r,cross2_r,cross3_r,cross1_z_r,cross2_z_r,cross3_z_r,p_c,div_c,nx,ny,nz1,nz2,Nz,L_x,L_y,L_z,nt,tau,l,0);
        #pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j <(ny/2+1); j++) {
                for (int k = 0; k < nz1; k++) {
                    int index = (i * (ny / 2 + 1) + j) * nz1 + k;
                    k1_V1[index][0] = F1_c_[index][0];
                    k1_V1[index][1] = F1_c_[index][1];
                    F1_c_[index][0] = V1_c_[index][0] + tau / 2 * k1_V1[index][0];
                    F1_c_[index][1] = V1_c_[index][1] + tau / 2 * k1_V1[index][1];

                    k1_V2[index][0] = F2_c_[index][0];
                    k1_V2[index][1] = F2_c_[index][1];
                    F2_c_[index][0] = V2_c_[index][0] + tau / 2 * k1_V2[index][0];
                    F2_c_[index][1] = V2_c_[index][1] + tau / 2 * k1_V2[index][1];
                }
                for (int k = 0; k < nz2; k++) {
                    int index = (i*(ny/2+1)+j)*nz2+k;
                    k1_V3[index][0] = F3_c_[index][0];
                    k1_V3[index][1] = F3_c_[index][1];
                    F3_c_[index][0] = V3_c_[index][0] + tau / 2 * k1_V3[index][0];
                    F3_c_[index][1] = V3_c_[index][1] + tau / 2 * k1_V3[index][1];
                }
            }
        }

        compute_F(F1_c_,F2_c_,F3_c_,f1_c_,f2_c_,f3_c_,f1_z_r_,f2_z_r_,f3_z_r_,f1_r_,f2_r_,f3_r_,V1_xy_r,V2_xy_r,V3_xy_r,rot1_c,rot2_c,rot3_c,rot1_r,rot2_r,rot3_r,rot1_xy_r,rot2_xy_r,rot3_xy_r,
                            cross1_c_,cross2_c_,cross3_c_,cross1_r,cross2_r,cross3_r,cross1_z_r,cross2_z_r,cross3_z_r,p_c,div_c,nx,ny,nz1,nz2,Nz,L_x,L_y,L_z,nt,tau,l,tau/2);
        #pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j <(ny/2+1); j++) {
                for (int k = 0; k < nz1; k++) {
                    int index = (i * (ny / 2 + 1) + j) * nz1 + k;
                    k2_V1[index][0] = F1_c_[index][0];
                    k2_V1[index][1] = F1_c_[index][1];
                    F1_c_[index][0] = V1_c_[index][0] + tau / 2 * k2_V1[index][0];
                    F1_c_[index][1] = V1_c_[index][1] + tau / 2 * k2_V1[index][1];

                    k2_V2[index][0] = F2_c_[index][0];
                    k2_V2[index][1] = F2_c_[index][1];
                    F2_c_[index][0] = V2_c_[index][0] + tau / 2 * k2_V2[index][0];
                    F2_c_[index][1] = V2_c_[index][1] + tau / 2 * k2_V2[index][1];
                }
                for (int k = 0; k < nz2; k++) {
                    int index = (i*(ny/2+1)+j)*nz2+k;
                    k2_V3[index][0] = F3_c_[index][0];
                    k2_V3[index][1] = F3_c_[index][1];
                    F3_c_[index][0] = V3_c_[index][0] + tau / 2 * k2_V3[index][0];
                    F3_c_[index][1] = V3_c_[index][1] + tau / 2 * k2_V3[index][1];
                }
            }
        }

        compute_F(F1_c_,F2_c_,F3_c_,f1_c_,f2_c_,f3_c_,f1_z_r_,f2_z_r_,f3_z_r_,f1_r_,f2_r_,f3_r_,V1_xy_r,V2_xy_r,V3_xy_r,rot1_c,rot2_c,rot3_c,rot1_r,rot2_r,rot3_r,rot1_xy_r,rot2_xy_r,rot3_xy_r,
                            cross1_c_,cross2_c_,cross3_c_,cross1_r,cross2_r,cross3_r,cross1_z_r,cross2_z_r,cross3_z_r,p_c,div_c,nx,ny,nz1,nz2,Nz,L_x,L_y,L_z,nt,tau,l,tau/2);
        #pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j <(ny/2+1); j++) {
                for (int k = 0; k < nz1; k++) {
                    int index = (i * (ny / 2 + 1) + j) * nz1 + k;
                    k3_V1[index][0] = F1_c_[index][0];
                    k3_V1[index][1] = F1_c_[index][1];
                    F1_c_[index][0] = V1_c_[index][0] + tau * k3_V1[index][0];
                    F1_c_[index][1] = V1_c_[index][1] + tau * k3_V1[index][1];

                    k3_V2[index][0] = F2_c_[index][0];
                    k3_V2[index][1] = F2_c_[index][1];
                    F2_c_[index][0] = V2_c_[index][0] + tau * k3_V2[index][0];
                    F2_c_[index][1] = V2_c_[index][1] + tau * k3_V2[index][1];
                }
                for (int k = 0; k < nz2; k++) {
                    int index = (i*(ny/2+1)+j)*nz2+k;
                    k3_V3[index][0] = F3_c_[index][0];
                    k3_V3[index][1] = F3_c_[index][1];
                    F3_c_[index][0] = V3_c_[index][0] + tau * k3_V3[index][0];
                    F3_c_[index][1] = V3_c_[index][1] + tau * k3_V3[index][1];
                }
            }
        }

        compute_F(F1_c_,F2_c_,F3_c_,f1_c_,f2_c_,f3_c_,f1_z_r_,f2_z_r_,f3_z_r_,f1_r_,f2_r_,f3_r_,V1_xy_r,V2_xy_r,V3_xy_r,rot1_c,rot2_c,rot3_c,rot1_r,rot2_r,rot3_r,rot1_xy_r,rot2_xy_r,rot3_xy_r,
                            cross1_c_,cross2_c_,cross3_c_,cross1_r,cross2_r,cross3_r,cross1_z_r,cross2_z_r,cross3_z_r,p_c,div_c,nx,ny,nz1,nz2,Nz,L_x,L_y,L_z,nt,tau,l,tau);
        #pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j <(ny/2+1); j++) {
                for (int k = 0; k < nz1; k++) {
                    int index = (i * (ny / 2 + 1) + j) * nz1 + k;
                    k4_V1[index][0] = F1_c_[index][0];
                    k4_V1[index][1] = F1_c_[index][1];

                    k4_V2[index][0] = F2_c_[index][0];
                    k4_V2[index][1] = F2_c_[index][1];
                }
                for (int k = 0; k < nz2; k++) {
                    int index = (i*(ny/2+1)+j)*nz2+k;
                    k4_V3[index][0] = F3_c_[index][0];
                    k4_V3[index][1] = F3_c_[index][1];
                }
            }
        }
        #pragma omp parallel for
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j <(ny/2+1); j++) {
                for (int k = 0; k < nz1; k++) {
                    int index = (i * (ny / 2 + 1) + j) * nz1 + k;
                    V1_c_[index][0] = V1_c_[index][0] + tau/6 *(k1_V1[index][0] + 2*k2_V1[index][0] + 2*k3_V1[index][0] + k4_V1[index][0]);
                    V1_c_[index][1] = V1_c_[index][1] + tau/6 *(k1_V1[index][1] + 2*k2_V1[index][1] + 2*k3_V1[index][1] + k4_V1[index][1]);
                    V2_c_[index][0] = V2_c_[index][0] + tau/6 *(k1_V2[index][0] + 2*k2_V2[index][0] + 2*k3_V2[index][0] + k4_V2[index][0]);
                    V2_c_[index][1] = V2_c_[index][1] + tau/6 *(k1_V2[index][1] + 2*k2_V2[index][1] + 2*k3_V2[index][1] + k4_V2[index][1]);
                }
                for (int k = 0; k < nz2; k++) {
                    int index = (i*(ny/2+1)+j)*nz2+k;
                    V3_c_[index][0] = V3_c_[index][0] + tau/6 *(k1_V3[index][0] + 2*k2_V3[index][0] + 2*k3_V3[index][0] + k4_V3[index][0]);
                    V3_c_[index][1] = V3_c_[index][1] + tau/6 *(k1_V3[index][1] + 2*k2_V3[index][1] + 2*k3_V3[index][1] + k4_V3[index][1]);
                }
            }
        }
    }
    fftw_free(k1_V1);
    fftw_free(k2_V1);
    fftw_free(k3_V1);
    fftw_free(k4_V1);
    fftw_free(k1_V2);
    fftw_free(k2_V2);
    fftw_free(k3_V2);
    fftw_free(k4_V2);
    fftw_free(k1_V3);
    fftw_free(k2_V3);
    fftw_free(k3_V3);
    fftw_free(k4_V3);
}

void NavierStokes(fftw_complex* V1_start, fftw_complex* V2_start, fftw_complex* V3_start, fftw_complex* p_c
                ,double* V1_r_,double* V2_r_,double* V3_r_
                ,int nx, int ny, int nz1, int nz2,int Nz,double L_x, double L_y, double L_z, int nt, double tau)
{   
    int i, j, k, index,k_x,k_y,k_z,index1,index2;
    
    fftw_complex *rot1_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz2);
    fftw_complex *rot2_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz2);
    fftw_complex *rot3_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    double * rot1_xy_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    double * rot2_xy_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    double * rot3_xy_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double * rot1_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    double * rot2_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    double * rot3_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double * V1_xy_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double * V2_xy_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double * V3_xy_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    double * cross1_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double * cross2_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double * cross3_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    double *cross1_z_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *cross2_z_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *cross3_z_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    fftw_complex *cross1_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *cross2_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *cross3_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz2);
    
    // f
    double *f1_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *f2_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *f3_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    
    double *f1_z_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *f2_z_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *f3_z_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    
    fftw_complex *F1_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *F2_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *F3_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz2);

    fftw_complex *f1_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *f2_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *f3_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz2);

    fftw_complex *div_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
       
    rungeKutta(V1_start,V2_start,V3_start,F1_c,F2_c,F3_c,f1_c,f2_c,f3_c,f1_z_r,f2_z_r,f3_z_r,f1_r,f2_r,f3_r,V1_xy_r,V2_xy_r,V3_xy_r,rot1_c,rot2_c,rot3_c,rot1_r,rot2_r,rot3_r,rot1_xy_r,rot2_xy_r,rot3_xy_r,
                            cross1_c,cross2_c,cross3_c,cross1_r,cross2_r,cross3_r,cross1_z_r,cross2_z_r,cross3_z_r,p_c,div_c,nx,ny,nz1,nz2,Nz,L_x,L_y,L_z,nt,tau);
    
    fftw_free(rot1_c);
    fftw_free(rot2_c);
    fftw_free(rot3_c);
    fftw_free(rot1_xy_r);
    fftw_free(rot2_xy_r);
    fftw_free(rot3_xy_r);
    fftw_free(rot1_r);
    fftw_free(rot2_r);
    fftw_free(rot3_r);
    fftw_free(V1_xy_r);
    fftw_free(V2_xy_r);
    fftw_free(V3_xy_r);
    fftw_free(cross1_r);
    fftw_free(cross2_r);
    fftw_free(cross3_r);
    fftw_free(cross1_z_r);
    fftw_free(cross2_z_r);
    fftw_free(cross3_z_r);
    fftw_free(cross1_c);
    fftw_free(cross2_c);
    fftw_free(cross3_c);
    fftw_free(f1_r);
    fftw_free(f2_r);
    fftw_free(f3_r);
    fftw_free(f1_z_r);
    fftw_free(f2_z_r);
    fftw_free(f3_z_r);
    fftw_free(div_c);
}

int main() {
    int nx = 64; 
    int ny = 64; 
    int Nz = 64;
    int nz1 = Nz/2+1;
    int nz2 = Nz/2-1;
    double L_x = 2*M_PI, L_y = 2*M_PI, L_z = 2*M_PI;

    int nt = 3000;
    double T = 1;
    double tau = T / 3000; 

    int i, j, k, index;

    int max_threads = omp_get_max_threads();
    cout << "num threads = " << max_threads << endl;
    fftw_init_threads();
    fftw_plan_with_nthreads(max_threads);
    omp_set_num_threads(max_threads);

    // V
    double *V1_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *V2_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *V3_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    
    double *V1_z_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *V2_z_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *V3_z_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);

    fftw_complex *V1_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *V2_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *V3_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz2);
    
    // V_out 
    fftw_complex *V1_out_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *V2_out_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *V3_out_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz2);

    double * V1_out_xy_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double * V2_out_xy_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double * V3_out_xy_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);

    double *V1_out_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *V2_out_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *V3_out_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);

    // p
    double *p_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *p_z_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    fftw_complex *p_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);

    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz1; k++) {
                int index = (i*ny+j)*nz1+k;
                V1_r[index] = func_V1(i*L_x/nx, j*L_y/ny, k*L_z/Nz,0);
                V2_r[index] = func_V2(i*L_x/nx, j*L_y/ny, k*L_z/Nz,0);
                p_r[index] = func_p(i*L_x/nx, j*L_y/ny, k*L_z/Nz,0);
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 1; k <= nz2; k++) {
                int index = (i*ny+j)*nz2+k-1;
                V3_r[index] = func_V3(i*L_x/nx, j*L_y/ny, k*L_z/Nz,0);
            }           
        }
    }
    
    //
    in = V1_r; in_z = V1_z_r; complex_out = V1_c;
    initialize_fwd_r2r_cos(nx, ny, nz1, nz2);   
    initialize_fwd_r2c_cos(nx, ny, nz1, nz2);  
    in = V3_r; in_z = V3_z_r; complex_out = V3_c;
    initialize_fwd_r2r_sin(nx, ny, nz1, nz2);   
    initialize_fwd_r2c_sin(nx, ny, nz1, nz2); 
    complex_out = V1_c; re_out_xy = V1_out_xy_r; re_out = V1_out_r;
    initialize_bwd_c2r_cos(nx, ny, nz1, nz2);   
    initialize_bwd_r2r_cos(nx, ny, nz1, nz2);  
    complex_out = V3_c; re_out_xy = V3_out_xy_r; re_out = V3_out_r;
    initialize_bwd_c2r_sin(nx, ny, nz1, nz2);   
    initialize_bwd_r2r_sin(nx, ny, nz1, nz2); 

    //
    fftw_execute_r2r(plan_fwd_r2r_cos, V1_r, V1_z_r);
    fftw_execute_dft_r2c(plan_fwd_r2c_cos, V1_z_r, V1_c);

    fftw_execute_r2r(plan_fwd_r2r_cos, V2_r, V2_z_r);
    fftw_execute_dft_r2c(plan_fwd_r2c_cos, V2_z_r, V2_c);

    fftw_execute_r2r(plan_fwd_r2r_sin, V3_r, V3_z_r);
    fftw_execute_dft_r2c(plan_fwd_r2c_sin, V3_z_r, V3_c);

    // p
    fftw_execute_r2r(plan_fwd_r2r_cos, p_r, p_z_r);
    fftw_execute_dft_r2c(plan_fwd_r2c_cos, p_z_r, p_c);

    // normalization
    normalization(V1_c, nx,(ny/2+1), nz1, nx*ny*Nz);
    normalization(V2_c, nx,(ny/2+1), nz1, nx*ny*Nz);
    normalization(V3_c, nx,(ny/2+1), nz2, nx*ny*Nz);
    normalization(p_c, nx,(ny/2+1), nz1, nx*ny*Nz);

    double start = omp_get_wtime();
    NavierStokes(V1_c,V2_c,V3_c,p_c,V1_r,V2_r,V3_r,nx,ny,nz1,nz2,Nz,L_x,L_y,L_z,nt,tau);
    double end = omp_get_wtime();
    cout << setprecision(3) <<"time parallel = " << (end - start) << "s" << endl;
    
    fftw_execute_dft_c2r(plan_bwd_c2r_cos,V1_c,V1_out_xy_r);
    fftw_execute_r2r(plan_bwd_r2r_cos, V1_out_xy_r, V1_out_r);

    fftw_execute_dft_c2r(plan_bwd_c2r_cos,V2_c,V2_out_xy_r);
    fftw_execute_r2r(plan_bwd_r2r_cos, V2_out_xy_r, V2_out_r);
    
    fftw_execute_dft_c2r(plan_bwd_c2r_sin,V3_c,V3_out_xy_r);
    fftw_execute_r2r(plan_bwd_r2r_sin, V3_out_xy_r, V3_out_r);

    double err1 = 0.0, err2 = 0.0, err3= 0.0, err1_, err2_, err3_;

    #pragma omp parallel for
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz1; k++) {
                int index1 = (i*ny+j)*nz1+k;
                double err1_ = fabs(V1_out_r[index1] - func_V1(i*L_x/nx, j*L_y/ny, k*L_z/Nz,nt*tau));
                double err2_ = fabs(V2_out_r[index1] - func_V2(i*L_x/nx, j*L_y/ny, k*L_z/Nz,nt*tau));
                err1 += err1_ * err1_;
                err2 += err2_ * err2_;
            }
            for (int k = 1; k <= nz2; k++) {
                int index2 = (i*ny+j)*nz2+k-1;
                double err3_ = fabs(V3_out_r[index2] - func_V3(i*L_x/nx, j*L_y/ny, k*L_z/Nz,nt*tau));
                err3 += err3_ * err3_;           
            }     
        }  
    }
    cout << "1 = " << sqrt(err1) << endl;
    cout << "2 = " << sqrt(err2) << endl;
    cout << "3 = " << sqrt(err3) << endl;
    
    
    // для даления планов
    finalize_fft_plans();
    fftw_cleanup_threads();
    
    fftw_free(V1_r);
    fftw_free(V2_r);
    fftw_free(V3_r);
    fftw_free(V1_z_r);
    fftw_free(V2_z_r);
    fftw_free(V3_z_r);
    fftw_free(V1_c);
    fftw_free(V2_c);
    fftw_free(V3_c);
    fftw_free(V1_out_c);
    fftw_free(V2_out_c);
    fftw_free(V3_out_c);
    fftw_free(V1_out_xy_r);
    fftw_free(V2_out_xy_r);
    fftw_free(V3_out_xy_r);
    fftw_free(V1_out_r);
    fftw_free(V2_out_r);
    fftw_free(V3_out_r);
    fftw_free(p_r);
    fftw_free(p_z_r);
    fftw_free(p_c);
    return 0;
}

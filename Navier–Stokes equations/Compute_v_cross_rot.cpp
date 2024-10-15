#include <iostream>
#include <fftw3.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>

using namespace std;

double V1(double x, double y, double z) {
    return sin(4*x)* cos(6*y)*cos(8*M_PI*z);
}
double V2(double x, double y, double z) {
    return sin(2*x)* cos(2*y)*cos(2 *M_PI* z);
}
double V3(double x, double y, double z) {
    return sin(2*x)* cos(2*y)*sin(2 *M_PI* z);
}

double rot1(double x, double y, double z) {
    return -2* sin(2*x)* sin(2*y)*sin(2 *M_PI* z) + 2 * M_PI* sin(2*x)* cos(2*y)*sin(2 *M_PI* z) ;
}

double rot2(double x, double y, double z) {
    return -8 * M_PI * sin(4 * x) * cos(6 * y) * sin(8 * M_PI * z) - 2 * cos(2 * x) * cos(2 * y) * sin(2 * M_PI * z);
}

double rot3(double x, double y, double z) {
    return 2*cos(2*x)* cos(2*y)*cos(2 *M_PI* z) -(-6*sin(4*x)*sin(6*y)*cos(8*M_PI*z));
}

double divV(double x, double y, double z) {
    return 4*cos(4*x)*cos(6*y)*cos(8*M_PI*z)-2*sin(2*x)* sin(2*y)*cos(2 *M_PI*z)+ 2*M_PI*sin(2*x)* cos(2*y)*cos(2 *M_PI*z);
}

void normalization(fftw_complex* V1_hat, fftw_complex* V2_hat, fftw_complex* V3_hat, int nx, int ny, int nz)
{   
    int i, j, k, index;
    for(i = 0; i < nx; ++i) {
        for(j = 0; j < (ny/2+1); ++j) {
            for(k = 0; k < (nz/2+1); ++k) {
                index = (i*(ny/2+1)+j)*(nz/2+1)+k;
                //Normalization
                V1_hat[index][0] /= (nx * ny * nz);
                V1_hat[index][1] /= (nx * ny * nz);
                V2_hat[index][0] /= (nx * ny * nz);
                V2_hat[index][1] /= (nx * ny * nz);
            }
        }
    }
    for(int i = 0; i < nx; ++i) {
        for(int j = 0; j < (ny/2+1); ++j) {
            for(int k = 0; k < (nz/2-1); ++k) {
                int index = (i*(ny/2+1)+j)*(nz/2-1)+k;
                //Normalization
                V3_hat[index][0] /= (nx * ny * nz);
                V3_hat[index][1] /= (nx * ny * nz);
            }
        }
    }
}

void compute_rot(fftw_complex* V1_hat, fftw_complex* V2_hat, fftw_complex* V3_hat
                ,fftw_complex* Rot1_hat, fftw_complex* Rot2_hat, fftw_complex* Rot3_hat, int nx, int ny, int nz, int L_x, int L_y, int L_z)
{
    int i, j, k, index,k_x,k_y,k_z,index1,index2;
    double alpha = 2*M_PI/L_z;
    for(i = 0; i < nx; ++i) {
        for(j = 0; j < (ny/2+1); ++j) {
            for(k = 0; k < (nz/2+1); ++k) {
                index = (i * (ny/2+1) + j) * (nz/2+1) + k;

                k_x = i <= nx/2 ? i : i -nx;
                k_y = j <= ny/2 ? j : j -ny;
                k_z = k;

                // Rotational:
                Rot3_hat[index][0] = -(V2_hat[index][1] * k_x - V1_hat[index][1] * k_y); // dV2 / dx - dv1 / dy 
                Rot3_hat[index][1] = V2_hat[index][0] * k_x - V1_hat[index][0] * k_y;    // dV2 / dx - dv1 / dy    
            }
        }
    }

    for(i = 0; i < nx; ++i) {
        for(j = 0; j < (ny/2+1); ++j) {
            for(k = 0; k < (nz/2-1); ++k) {
                index1 = (i * (ny/2+1) + j) * (nz/2+1) + k + 1;
                index2 = (i * (ny/2+1) + j) * (nz/2-1) + k;

                k_x = i <= nx/2 ? i : i -nx;
                k_y = j <= ny/2 ? j : j -ny;
                k_z = k;

                // Rotational:
                Rot1_hat[index2][0] = (-V3_hat[index2][1] * (k_y)) -(-V2_hat[index1][0] * (k_z+1)*alpha);  // dV3 / dy - dv2 / dz 
                Rot1_hat[index2][1] = (V3_hat[index2][0] * (k_y)) - (-V2_hat[index1][1] * (k_z+1) *alpha); // dV3 / dy - dv2 / dz
                Rot2_hat[index2][0] = (-V1_hat[index1][0] * (k_z+1)*alpha) - (-V3_hat[index2][1] * (k_x)); // dV2 / dz - dv3 / dy
                Rot2_hat[index2][1] = (-V1_hat[index1][1] * (k_z+1)*alpha) - (V3_hat[index2][0] * (k_x));  // dV2 / dz - dv3 / dy
            }
        }
    }
}

void compute_div(fftw_complex* V1_hat, fftw_complex* V2_hat, fftw_complex* V3_hat
                ,fftw_complex* div_hat, int nx, int ny, int nz,int L_x, int L_y, int L_z)
{
    int i, j, k, index,k_x,k_y,k_z,index1,index2;
    double alpha = 2*M_PI/L_z;
    for(i = 0; i < nx; ++i) {
        for(j = 0; j < (ny/2+1); ++j) {
            k_x = i <= nx/2 ? i : i -nx;
            k_y = j <= ny/2 ? j : j -ny;
            for(k = 1; k < (nz/2); ++k) {
                index1 = (i * (ny/2+1) + j) * (nz/2+1) + k;
                index2 = (i * (ny/2+1) + j) * (nz/2-1) + k-1;
                k_z = k;
                // Divergence:
                div_hat[index1][0] = (-V1_hat[index1][1] * k_x) + (-V2_hat[index1][1] * k_y) + (V3_hat[index2][0] * k_z*alpha) ;
                div_hat[index1][1] = (V1_hat[index1][0] * k_x) +  (V2_hat[index1][0] * k_y ) + (V3_hat[index2][1] * k_z*alpha); 
            }
            // k = 0
            index = (i * (ny/2+1) + j) * (nz/2+1) + 0;
            div_hat[index][0] = (-V1_hat[index][1] * k_x) + (-V2_hat[index][1] * k_y);
            div_hat[index][1] = (V1_hat[index][0] * k_x) +  (V2_hat[index][0] * k_y );

            // k = nz/2
            index = (i * (ny/2+1) + j) * (nz/2+1) + nz/2;
            div_hat[index][0] = (-V1_hat[index][1] * k_x) + (-V2_hat[index][1] * k_y);
            div_hat[index][1] = (V1_hat[index][0] * k_x) +  (V2_hat[index][0] * k_y );
        }
    }
}

void compute_v_cross_rot(double *V1, double *V2, double *V3, double *rotv1, double *rotv2, double *rotv3
                ,double *v_cross_rot1, double *v_cross_rot2, double *v_cross_rot3, int nx, int ny, int nz1, int nz2)
{
    int index1,index2;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz1; k++) {
                index1 = (i*ny+j)*nz1+k;
                index2 = (i*ny+j)*nz2+k-1;
                if (k != 0 && k != nz1 - 1) 
                {
                    
                    v_cross_rot1[index1] = V2[index1]*rotv3[index1] - V3[index2]*rotv2[index2];
                    v_cross_rot2[index1] = V3[index2]*rotv1[index2] - V1[index1]*rotv3[index1];
                    v_cross_rot3[index2] = V1[index1]*rotv2[index2] - V2[index1]*rotv1[index2];
                } else 
                {
                    v_cross_rot1[index1] = V2[index1]*rotv3[index1] - 0;
                    v_cross_rot2[index1] = 0 - V1[index1]*rotv3[index1];
                }
            }      
        }
    }
}

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

int main() {
    int nx = 16; 
    int ny = 16; 
    int Nz = 16;
    int nz1 = Nz/2+1;
    int nz2 = Nz/2-1;
    double L_x = 2*M_PI, L_y = 2*M_PI, L_z = 1;

    int i, j, k, index;

    double *re_V1 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *re_V2 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *re_V3 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    
    double *re_V1_z = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *re_V2_z = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *re_V3_z = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);

    double * v_cross_rot_1 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double * v_cross_rot_2 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double * v_cross_rot_3 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);

    double *re_v_cross_rot_1_z = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *re_v_cross_rot_2_z = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double *re_v_cross_rot_3_z = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
  
    fftw_complex *complex_V1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *complex_V2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *complex_V3 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz2);

    fftw_complex *complex_v_cross_rot_1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *complex_v_cross_rot_2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *complex_v_cross_rot_3 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz2);
    
    fftw_complex *complex_Rot1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz2);
    fftw_complex *complex_Rot2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz2);
    fftw_complex *complex_Rot3 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *complex_div = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) *  nz1);

    double * computedRotV1_xy = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    double * computedRotV2_xy = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    double * computedRotV3_xy = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double * computedDivV_xy = (double*) fftw_malloc(sizeof(double) * nx * ny *  nz1);

    double * computedRotV1 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    double * computedRotV2 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz2);
    double * computedRotV3 = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);
    double * computedDivV = (double*) fftw_malloc(sizeof(double) * nx * ny *  nz1);

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz1; k++) {
                index = (i*ny+j)*nz1+k;
                re_V1[index] = V1(i*L_x/nx, j*L_y/ny, k*L_z/Nz);
                re_V2[index] = V2(i*L_x/nx, j*L_y/ny, k*L_z/Nz);
            }
        }
    }

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 1; k <= nz2; k++) {
                index = (i*ny+j)*nz2+k-1;
                re_V3[index] = V3(i*L_x/nx, j*L_y/ny, k*L_z/Nz);
            }           
        }
    }

    //Инициализация планов
    in = re_V1; in_z = re_V1_z; complex_out = complex_V1;
    initialize_fwd_r2r_cos(nx, ny, nz1, nz2);   
    initialize_fwd_r2c_cos(nx, ny, nz1, nz2);    
    fftw_execute(plan_fwd_r2r_cos);
    fftw_execute(plan_fwd_r2c_cos);

    in = re_V2; in_z = re_V2_z; complex_out = complex_V2;
    initialize_fwd_r2r_cos(nx, ny, nz1, nz2);   
    initialize_fwd_r2c_cos(nx, ny, nz1, nz2); 
    fftw_execute(plan_fwd_r2r_cos);
    fftw_execute(plan_fwd_r2c_cos);

    in = re_V3; in_z = re_V3_z; complex_out = complex_V3;
    initialize_fwd_r2r_sin(nx, ny, nz1, nz2);   
    initialize_fwd_r2c_sin(nx, ny, nz1, nz2); 
    fftw_execute(plan_fwd_r2r_sin);
    fftw_execute(plan_fwd_r2c_sin);

    normalization(complex_V1, complex_V2, complex_V3,nx, ny, Nz);
    // Compute rot and div V
    compute_rot(complex_V1, complex_V2, complex_V3, complex_Rot1, complex_Rot2, complex_Rot3, nx, ny, Nz,L_x,L_y,L_z);
    compute_div(complex_V1, complex_V2, complex_V3, complex_div, nx, ny, Nz,L_x,L_y,L_z);

    // Backward transformation
    complex_out = complex_Rot1; re_out_xy = computedRotV1_xy; re_out = computedRotV1;
    initialize_bwd_c2r_sin(nx, ny, nz1, nz2);   
    initialize_bwd_r2r_sin(nx, ny, nz1, nz2); 
    fftw_execute(plan_bwd_c2r_sin);
    fftw_execute(plan_bwd_r2r_sin);

    complex_out = complex_Rot2; re_out_xy = computedRotV2_xy; re_out = computedRotV2;
    initialize_bwd_c2r_sin(nx, ny, nz1, nz2);   
    initialize_bwd_r2r_sin(nx, ny, nz1, nz2); 
    fftw_execute(plan_bwd_c2r_sin);
    fftw_execute(plan_bwd_r2r_sin);

    complex_out = complex_Rot3; re_out_xy = computedRotV3_xy; re_out = computedRotV3;
    initialize_bwd_c2r_cos(nx, ny, nz1, nz2);   
    initialize_bwd_r2r_cos(nx, ny, nz1, nz2);  
    fftw_execute(plan_bwd_c2r_cos);
    fftw_execute(plan_bwd_r2r_cos);
    
    // Посчитал rotV1 rotV2 rotV3 с помощью преобразование Фурье
    // Посчитаем V x rotV 
    compute_v_cross_rot(re_V1,re_V2,re_V3,computedRotV1,computedRotV2,computedRotV3,v_cross_rot_1,v_cross_rot_2,v_cross_rot_3,nx,ny, nz1,nz2);

    // сравнил полученые значения V x rotV с точными, убедится, что мы получаем правильные результаты
    double err_v_cross_rot1 = 0.0, err_v_cross_rot2 = 0.0, err_v_cross_rot3 = 0.0, err_v_cross_rot1_, err_v_cross_rot2_, err_v_cross_rot3_;
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz1; k++) {
                int index1 = (i*ny+j)*nz1+k;
                err_v_cross_rot1_ = fabs(v_cross_rot_1[index1] - (V2(i*L_x/nx, j*L_y/ny, k*L_z/Nz)*rot3(i*L_x/nx, j*L_y/ny, k*L_z/Nz)-  V3(i*L_x/nx, j*L_y/ny, k*L_z/Nz)*rot2(i*L_x/nx, j*L_y/ny, k*L_z/Nz)));
                err_v_cross_rot2_ = fabs(v_cross_rot_2[index1] - (V3(i*L_x/nx, j*L_y/ny, k*L_z/Nz)*rot1(i*L_x/nx, j*L_y/ny, k*L_z/Nz)-  V1(i*L_x/nx, j*L_y/ny, k*L_z/Nz)*rot3(i*L_x/nx, j*L_y/ny, k*L_z/Nz)));
                err_v_cross_rot1 += err_v_cross_rot1_ * err_v_cross_rot1_;
                err_v_cross_rot2 += err_v_cross_rot2_ * err_v_cross_rot2_;
            }
            for (k = 1; k <= nz2; k++) {
                int index2 = (i*ny+j)*nz2+k-1;
                err_v_cross_rot3_ = fabs(v_cross_rot_3[index2] - (V1(i*L_x/nx, j*L_y/ny, k*L_z/Nz)*rot2(i*L_x/nx, j*L_y/ny, k*L_z/Nz)-  V2(i*L_x/nx, j*L_y/ny, k*L_z/Nz)*rot1(i*L_x/nx, j*L_y/ny, k*L_z/Nz)));
                err_v_cross_rot3 += err_v_cross_rot3_ * err_v_cross_rot3_;           
            }     
        }  
    }
    cout << "err_v_cross_rot1 = " << sqrt(err_v_cross_rot1) << endl;
    cout << "err_v_cross_rot2 = " << sqrt(err_v_cross_rot2) << endl;
    cout << "err_v_cross_rot3 = " << sqrt(err_v_cross_rot3) << endl;
    
    // Forward transformation  
    // Сделал fft для V x rotV, 
    // их коэффициенты пространства фурье сохранились в массиве complex_v_cross_rot_1,2,3...
    // наверху показано, что полученые значения V x rotV точные, поэтому здесь счтаемся их коэффициенты тоже точные

    in = v_cross_rot_1; in_z = re_v_cross_rot_1_z; complex_out = complex_v_cross_rot_1;
    initialize_fwd_r2r_cos(nx, ny, nz1, nz2);   
    initialize_fwd_r2c_cos(nx, ny, nz1, nz2);    
    fftw_execute(plan_fwd_r2r_cos);
    fftw_execute(plan_fwd_r2c_cos);

    in = v_cross_rot_2; in_z = re_v_cross_rot_2_z; complex_out = complex_v_cross_rot_2;
    initialize_fwd_r2r_cos(nx, ny, nz1, nz2);   
    initialize_fwd_r2c_cos(nx, ny, nz1, nz2); 
    fftw_execute(plan_fwd_r2r_cos);
    fftw_execute(plan_fwd_r2c_cos);

    in = v_cross_rot_3; in_z = re_v_cross_rot_3_z; complex_out = complex_v_cross_rot_3;
    initialize_fwd_r2r_sin(nx, ny, nz1, nz2);   
    initialize_fwd_r2c_sin(nx, ny, nz1, nz2); 
    fftw_execute(plan_fwd_r2r_sin);
    fftw_execute(plan_fwd_r2c_sin);   

    // для даления планов
    finalize_fft_plans();

    fftw_free(re_V1);
    fftw_free(re_V2);
    fftw_free(re_V3);
    
    fftw_free(re_V1_z);
    fftw_free(re_V2_z);
    fftw_free(re_V3_z);

    fftw_free(complex_V1);
    fftw_free(complex_V2);
    fftw_free(complex_V3);

    fftw_free(complex_Rot1);
    fftw_free(complex_Rot2);
    fftw_free(complex_Rot3);
    fftw_free(complex_div);

    fftw_free(computedRotV1_xy);
    fftw_free(computedRotV2_xy);
    fftw_free(computedRotV3_xy);
    fftw_free(computedDivV_xy);

    fftw_free(computedRotV1);
    fftw_free(computedRotV2);
    fftw_free(computedRotV3);
    fftw_free(computedDivV);
    
    fftw_free(v_cross_rot_1);
    fftw_free(v_cross_rot_2);
    fftw_free(v_cross_rot_3);
    
    fftw_free(re_v_cross_rot_1_z);
    fftw_free(re_v_cross_rot_2_z);
    fftw_free(re_v_cross_rot_3_z);
    
    fftw_free(complex_v_cross_rot_1);
    fftw_free(complex_v_cross_rot_2);
    fftw_free(complex_v_cross_rot_3);
    return 0;
}
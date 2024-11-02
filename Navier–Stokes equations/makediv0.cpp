#include <iostream>
#include <fftw3.h>
#include <math.h>
#include <complex.h>
#include <stdlib.h>

using namespace std;

double func_V1(double x, double y, double z) {
    return sin(4*x)* cos(6*y)*cos(8*M_PI*z);
}
double func_V2(double x, double y, double z) {
    return sin(2*x)* cos(2*y)*cos(2 *M_PI* z);
}
double func_V3(double x, double y, double z) {
    return sin(2*x)* cos(2*y)*sin(2 *M_PI* z);
}

double func_divV(double x, double y, double z) {
    return 4*cos(4*x)*cos(6*y)*cos(8*M_PI*z)-2*sin(2*x)* sin(2*y)*cos(2 *M_PI*z)+ 2*M_PI*sin(2*x)* cos(2*y)*cos(2 *M_PI*z);
}


void normalization(fftw_complex* V_c_, int nx, int ny, int nz, int factor)
{   
    int i, j, k, index;
    for(i = 0; i < nx; ++i) {
        for(j = 0; j < ny; ++j) {
            for(k = 0; k < nz; ++k) {
                index = (i*ny+j)*nz+k;
                //Normalization
                V_c_[index][0] /= factor;
                V_c_[index][1] /= factor;
            }
        }
    }
}

void compute_div(fftw_complex* V1_c_, fftw_complex* V2_c_, fftw_complex* V3_c_
                ,fftw_complex* div_c_, int nx, int ny, int nz,double L_x, double L_y, double L_z)
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
                div_c_[index1][0] = (-V1_c_[index1][1] * k_x) + (-V2_c_[index1][1] * k_y) + (V3_c_[index2][0] * k_z*alpha) ;
                div_c_[index1][1] = (V1_c_[index1][0] * k_x) +  (V2_c_[index1][0] * k_y ) + (V3_c_[index2][1] * k_z*alpha); 
            }
            // k = 0
            index = (i * (ny/2+1) + j) * (nz/2+1) + 0;
            div_c_[index][0] = (-V1_c_[index][1] * k_x) + (-V2_c_[index][1] * k_y);
            div_c_[index][1] = (V1_c_[index][0] * k_x) +  (V2_c_[index][0] * k_y );

            // k = nz/2
            index = (i * (ny/2+1) + j) * (nz/2+1) + nz/2;
            div_c_[index][0] = (-V1_c_[index][1] * k_x) + (-V2_c_[index][1] * k_y);
            div_c_[index][1] = (V1_c_[index][0] * k_x) +  (V2_c_[index][0] * k_y );
        }
    }
}

void make_div_0(fftw_complex* V1_old, fftw_complex* V2_old, fftw_complex* V3_old, fftw_complex* fi,
                fftw_complex* V1_new, fftw_complex* V2_new, fftw_complex* V3_new, int nx, int ny, int nz1, int nz2,double L_x, double L_y, double L_z)
{
    int i, j, k,k_x,k_y,k_z,index1,index2;
    double alpha = 2*M_PI/L_z;
    for(i = 0; i < nx; ++i) {
        for(j = 0; j < (ny/2+1); ++j) {
            for(k = 0; k < nz1; ++k) {
                k_x = i <= nx/2 ? i : i -nx;
                k_y = j <= ny/2 ? j : j -ny;
                k_z = k;
                index1 = (i * (ny/2+1) + j) * nz1 + k;
                index2 = (i * (ny/2+1) + j) * nz2 + k-1;

                if (i==0 && j==0 && k==0)
                {
                    fi[index1][0] = 0;
                    fi[index1][1] = 0;
                }
                else if (k != 0 && k != nz1 - 1) 
                {
                    fi[index1][0] = ((-V1_old[index1][1] * k_x) + (-V2_old[index1][1] * k_y) + (V3_old[index2][0] * k_z*alpha)) / -(k_x*k_x + k_y*k_y + k_z*k_z*alpha*alpha);
                    fi[index1][1] = ((V1_old[index1][0] * k_x) +  (V2_old[index1][0] * k_y ) + (V3_old[index2][1] * k_z*alpha)) / -(k_x*k_x + k_y*k_y + k_z*k_z*alpha*alpha); 
                } 
                else if (k == 0 || k == nz1 - 1) 
                {
                    fi[index1][0] = (-V1_old[index1][1] * k_x) + (-V2_old[index1][1] * k_y)/ -(k_x*k_x + k_y*k_y + k_z*k_z*alpha*alpha);
                    fi[index1][1] = (V1_old[index1][0] * k_x) +  (V2_old[index1][0] * k_y )/ -(k_x*k_x + k_y*k_y + k_z*k_z*alpha*alpha);
                }
            }
        }
    }

    for(i = 0; i < nx; ++i) {
        for(j = 0; j < (ny/2+1); ++j) {
            for(k = 0; k < nz1; ++k) {
                k_x = i <= nx/2 ? i : i -nx;
                k_y = j <= ny/2 ? j : j -ny;
                k_z = k;
                index1 = (i * (ny/2+1) + j) * nz1 + k;
                index2 = (i * (ny/2+1) + j) * nz2 + k-1;

                if (k != 0 && k != nz1 - 1) 
                {
                    V1_new[index1][0] = V1_old[index1][0] - (-fi[index1][1] * k_x);
                    V1_new[index1][1] = V1_old[index1][1] - (fi[index1][0] * k_x);
                    V2_new[index1][0] = V2_old[index1][0] - (-fi[index1][1] * k_y);
                    V2_new[index1][1] = V2_old[index1][1] - (fi[index1][0] * k_y);
                    V3_new[index2][0] = V3_old[index2][0] - (-fi[index1][0]* k_z*alpha);
                    V3_new[index2][1] = V3_old[index2][1] - (-fi[index1][1]* k_z*alpha);
                } 
                else if (k == 0 || k == nz1 - 1) 
                {
                    V1_new[index1][0] = V1_old[index1][0] - (-fi[index1][1] * k_x);
                    V1_new[index1][1] = V1_old[index1][1] - (fi[index1][0] * k_x);
                    V2_new[index1][0] = V2_old[index1][0] - (-fi[index1][1] * k_y);
                    V2_new[index1][1] = V2_old[index1][1] - (fi[index1][0] * k_y);
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

    fftw_complex *divV_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);

    double * divV_xy_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);

    double *divV_out_r = (double*) fftw_malloc(sizeof(double) * nx * ny * nz1);


    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz1; k++) {
                index = (i*ny+j)*nz1+k;
                V1_r[index] = func_V1(i*L_x/nx, j*L_y/ny, k*L_z/Nz);
                V2_r[index] = func_V2(i*L_x/nx, j*L_y/ny, k*L_z/Nz);
            }
        }
    }

    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 1; k <= nz2; k++) {
                index = (i*ny+j)*nz2+k-1;
                V3_r[index] = func_V3(i*L_x/nx, j*L_y/ny, k*L_z/Nz);
            }           
        }
    }
    
    // V
    in = V1_r; in_z = V1_z_r; complex_out = V1_c;
    initialize_fwd_r2r_cos(nx, ny, nz1, nz2);   
    initialize_fwd_r2c_cos(nx, ny, nz1, nz2);    
    fftw_execute(plan_fwd_r2r_cos);
    fftw_execute(plan_fwd_r2c_cos);

    in = V2_r; in_z = V2_z_r; complex_out = V2_c;
    initialize_fwd_r2r_cos(nx, ny, nz1, nz2);   
    initialize_fwd_r2c_cos(nx, ny, nz1, nz2); 
    fftw_execute(plan_fwd_r2r_cos);
    fftw_execute(plan_fwd_r2c_cos);

    in = V3_r; in_z = V3_z_r; complex_out = V3_c;
    initialize_fwd_r2r_sin(nx, ny, nz1, nz2);   
    initialize_fwd_r2c_sin(nx, ny, nz1, nz2); 
    fftw_execute(plan_fwd_r2r_sin);
    fftw_execute(plan_fwd_r2c_sin);

    // normalization
    normalization(V1_c, nx,(ny/2+1), nz1, nx*ny*Nz);
    normalization(V2_c, nx,(ny/2+1), nz1, nx*ny*Nz);
    normalization(V3_c, nx,(ny/2+1), nz2, nx*ny*Nz);


    compute_div(V1_c,V2_c,V3_c,divV_c,nx,ny,Nz,L_x,L_y,L_z);
    
    complex_out = divV_c; re_out_xy = divV_xy_r; re_out = divV_out_r;
    initialize_bwd_c2r_cos(nx, ny, nz1, nz2);   
    initialize_bwd_r2r_cos(nx, ny, nz1, nz2);  
    fftw_execute(plan_bwd_c2r_cos);
    fftw_execute(plan_bwd_r2r_cos);

    double err1 = 0.0, err2 = 0.0, err3= 0.0, err1_, err2_, err3_;
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz1; k++) {
                int index1 = (i*ny+j)*nz1+k;
                err1_ = fabs(fabs(divV_out_r[index1] - 0));
                err1 += err1_ * err1_;
            }
            
        }  
    }
    cout << "||div_Vold - 0||_2 = " << sqrt(err1) << endl;

    // V_out 
    fftw_complex *V1_div0_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *V2_div0_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    fftw_complex *V3_div0_c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz2);

    fftw_complex *fi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * (ny/2+1) * nz1);
    make_div_0(V1_c,V2_c,V3_c,fi,V1_div0_c,V2_div0_c,V3_div0_c,nx,ny,nz1,nz2,L_x,L_y,L_z);
    compute_div(V1_div0_c,V2_div0_c,V3_div0_c,divV_c,nx,ny,Nz,L_x,L_y,L_z);
    
    complex_out = divV_c; re_out_xy = divV_xy_r; re_out = divV_out_r;
    initialize_bwd_c2r_cos(nx, ny, nz1, nz2);   
    initialize_bwd_r2r_cos(nx, ny, nz1, nz2);  
    fftw_execute(plan_bwd_c2r_cos);
    fftw_execute(plan_bwd_r2r_cos);
    err1 = 0.0;
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz1; k++) {
                int index1 = (i*ny+j)*nz1+k;
                err1_ = fabs(divV_out_r[index1] - 0);
                err1 += err1_ * err1_;
            }
        }  
    }
    cout << "||div_Vnew - 0||_2 = " << sqrt(err1) << endl;
    
    // для даления планов
    finalize_fft_plans();

    fftw_free(V1_r);
    fftw_free(V2_r);
    fftw_free(V3_r);
    fftw_free(V1_z_r);
    fftw_free(V2_z_r);
    fftw_free(V3_z_r);
    fftw_free(V1_c);
    fftw_free(V2_c);
    fftw_free(V3_c);
    fftw_free(divV_c);
    fftw_free(divV_xy_r);
    fftw_free(divV_out_r);
    return 0;
}
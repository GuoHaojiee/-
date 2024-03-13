#include <iostream>
#include <fftw3.h>
#include <cmath>
#include <stdlib.h>
#include <string.h>

using namespace std;

double func(double x, double y, double z)
{
	return sin(x)*cos(y)*cos(2*M_PI*z);
}

double dfunc_xy(double x, double y, double z)
{
    return  -func(x,y,z);   
}

double dfunc_z(double x, double y, double z)
{
    return  -4*M_PI*M_PI*func(x,y,z);   
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
    int N1 = 8; 
    int N2 = 8; 
    int N3 = 8;
    int n3 = N3/2+1;
    int istride, ostride;
    int idist, odist;
    int howmany;
    int rank;
    double L_x = 2*M_PI, L_y = 2*M_PI, L_z = 1;
    double *in_ = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *in = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *in2 = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *in_dfunc_xy = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *in_dfunc_z = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *ino = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *out_z = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    fftw_complex *out = (fftw_complex*)fftw_malloc(N1*(N2/2+1)*n3 * sizeof(fftw_complex));

    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < N2; ++j) {
            for(int k = 0; k < n3; ++k) {
                in[(i*N2+j)*n3+k] = func(i*L_x/N1, j*L_y/N2, k*L_z/N3);
                in_[(i*N2+j)*n3+k] = func(i*L_x/N1, j*L_y/N2, k*L_z/N3);
                in_dfunc_xy[(i*N2+j)*n3+k] = dfunc_xy(i*L_x/N1, j*L_y/N2, k*L_z/N3);
                in_dfunc_z[(i*N2+j)*n3+k] = dfunc_z(i*L_x/N1, j*L_y/N2, k*L_z/N3);
            }
        }
    }

    // Forward transformation R -> R z
    rank = 1;
    int n[] = {n3};
    howmany = N1*N2;
    istride = 1; ostride = 1;
    idist = n3;  odist = n3;
    int *inembed = n, *onembed = n;
    const fftw_r2r_kind kind[] = {FFTW_REDFT00};

    fftw_plan plan_r2r_z = fftw_plan_many_r2r(rank, n, howmany,
                                              in, inembed, istride, idist,
                                              ino, onembed, ostride, odist,
                                              kind, FFTW_ESTIMATE);

    fftw_execute(plan_r2r_z);
    fftw_destroy_plan(plan_r2r_z);

    // Forward transformation R2 -> C2 x,y
    int nn[] = {N1, N2};
    int inembed2[] =  {N1, N2};
    int onembed2[] =  {N1, N2/2+1};
    istride = 1; ostride = 1;
    idist = N1*N2; odist = N1*(N2/2+1);

    fftw_plan fplan_3d = fftw_plan_many_dft_r2c(2, nn, n3,
                                                ino, inembed2, istride, idist,
                                                out, onembed2, ostride, odist,
                                                FFTW_ESTIMATE);

    fftw_execute(fplan_3d);
    fftw_destroy_plan(fplan_3d);

    //Normalization
    for (int i = 0; i < N1*(N2/2+1)*n3; ++i)
    {
        out[i][0] /= N1*N2;
        out[i][1] /= N1*N2;
    }
    
    // Backward transformation C2 -> R2 x,y
    inembed2[0] = N1; inembed2[1] = N2/2+1;
    onembed2[0] = N1; onembed2[1] = N2;
    istride = 1; ostride = 1;
    idist = N1*(N2/2+1); odist = N1*N2;
    fftw_plan bplan_3d = fftw_plan_many_dft_c2r(2, nn, n3,
                                                    out, inembed2, istride, idist,
                                                    in, onembed2, ostride, odist,
                                                    FFTW_ESTIMATE);
    fftw_execute(bplan_3d);
    fftw_destroy_plan(bplan_3d);
    
    //Normalization
    for (int i = 0; i < N1*N2*n3; ++i)
    {
        in[i] /= N3;
        in2[i] = in[i]; // save value in[]
    }

    // Backward transformation R -> R z
    rank = 1;
    n[0] = n3;
    howmany = N1 * N2;
    istride = 1; ostride = 1;
    idist = n3; odist = n3;
    int *inembed3 = n, *onembed3 = n;
    fftw_r2r_kind kind_z_inverse[] = {FFTW_REDFT00};
    fftw_plan plan_r2r_z_inverse = fftw_plan_many_r2r(rank, n, howmany,
                                                      in, inembed3, istride, idist,
                                                      out_z, onembed3, ostride, odist,
                                                      kind, FFTW_ESTIMATE);    
    // calculate d2f/dx2
    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < N2; ++j) {
            for(int k = 0; k < n3; ++k) {
                in[(i*N2+j)*n3+k] *= -k*k; // Здесь я не уверен!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                //Не знаю почему: если я сделал так double k_x = i <= nx/2 ? i : i - nx  результат неправильный
            }
        }
    }                                              
    fftw_execute(plan_r2r_z_inverse);
    cout << "err d2f/dx2 = " << err_calculation(out_z, in_dfunc_xy, N1*N2*n3) << endl;

    // calculate d2f/dy2
    for (int i = 0; i < N1*N2*n3; ++i)
    {
        in[i] = in2[i];
    }
    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < N2; ++j) {
            for(int k = 0; k < n3; ++k) {
                in[(i*N2+j)*n3+k] *= -k*k;// Здесь я не уверен!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                //Не знаю почему: если я сделал так double k_y = i <= nx/2 ? i : i - ny  результат неправильный
            }
        }
    }                                              
    fftw_execute(plan_r2r_z_inverse);
    cout << "err d2f/dy2 = " << err_calculation(out_z, in_dfunc_xy, N1*N2*n3) << endl;

    // calculate d2f/dz2
    for (int i = 0; i < N1*N2*n3; ++i)
    {
        in[i] = in2[i];
    }
    double alpha = 2*M_PI/L_z;
    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < N2; ++j) {
            for(int k = 0; k < n3; ++k) {
                in[(i*N2+j)*n3+k] *= -k*k*alpha*alpha;
            }
        }
    }   
    fftw_execute(plan_r2r_z_inverse);
    cout << "err d2f/dz2 = " << err_calculation(out_z, in_dfunc_z, N1*N2*n3) << endl;

    fftw_destroy_plan(plan_r2r_z_inverse);

    free(in);
    free(ino);
    free(out_z);
    free(out);
    free(in_);
    free(in2);
    free(in_dfunc_xy);
    free(in_dfunc_z);
    return 0;
}
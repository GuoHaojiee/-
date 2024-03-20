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
    double *in = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *ino = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *in_dfunc_xy = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *in_dfunc_z = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *d_out = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    fftw_complex *out = (fftw_complex*)fftw_malloc(N1*(N2/2+1)*n3 * sizeof(fftw_complex));

    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < N2; ++j) {
            for(int k = 0; k < n3; ++k) {
                int index = (i*N2+j)*n3+k;
                in[index] = func(i*L_x/N1, j*L_y/N2, k*L_z/N3);
                in_dfunc_xy[index] = dfunc_xy(i*L_x/N1, j*L_y/N2, k*L_z/N3);
                in_dfunc_z[index] = dfunc_z(i*L_x/N1, j*L_y/N2, k*L_z/N3);
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
    istride = n3; ostride = n3;
    idist = 1; odist = 1;

    fftw_plan fplan_3d = fftw_plan_many_dft_r2c(2, nn, n3,
                                                ino, inembed2, istride, idist,
                                                out, onembed2, ostride, odist,
                                                FFTW_ESTIMATE);

    fftw_execute(fplan_3d);
    fftw_destroy_plan(fplan_3d);

    fftw_complex *out_x = (fftw_complex*)fftw_malloc(N1*(N2/2+1)*n3 * sizeof(fftw_complex));
    fftw_complex *out_y = (fftw_complex*)fftw_malloc(N1*(N2/2+1)*n3 * sizeof(fftw_complex));
    fftw_complex *out_z = (fftw_complex*)fftw_malloc(N1*(N2/2+1)*n3 * sizeof(fftw_complex));
    
    //Normalization
    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < (N2/2+1); ++j) {
            for(int k = 0; k < n3; ++k) {
                int index = (i*(N2/2+1)+j)*n3+k;
                int k_x = i <= N1/2 ? i : i - N1;
                int k_y = j <= N2/2 ? j : j - N2;
                out[index][0] /= N1*N2*N3;
                out[index][1] /= N1*N2*N3;
                out_x[index][0] = out[index][0] * -(k_x)*(k_x); 
                out_x[index][1] = out[index][1] * -(k_x)*(k_x);
                
                out_y[index][0] = out[index][0] * -(k_y)*(k_y); 
                out_y[index][1] = out[index][1] * -(k_y)*(k_y);

                double alpha = 2*M_PI/L_z;
                out_z[index][0] = out[index][0] * -k*k*alpha*alpha;
                out_z[index][1] = out[index][1] * -k*k*alpha*alpha;
            }
        }
    }  
    
    // Backward transformation C2 -> R2 x,y
    inembed2[0] = N1; inembed2[1] = N2/2+1;
    onembed2[0] = N1; onembed2[1] = N2;
    istride = n3; ostride = n3;
    idist = 1; odist = 1;
    fftw_plan bplan_3d = fftw_plan_many_dft_c2r(2, nn, n3,
                                                    out, inembed2, istride, idist,
                                                    ino, onembed2, ostride, odist,
                                                    FFTW_ESTIMATE);
        
    // Backward transformation R -> R z
    rank = 1;
    n[0] = n3;
    howmany = N1 * N2;
    istride = 1; ostride = 1;
    idist = n3; odist = n3;
    int *inembed3 = n, *onembed3 = n;
    fftw_r2r_kind kind_z_inverse[] = {FFTW_REDFT00};
    fftw_plan plan_r2r_z_inverse = fftw_plan_many_r2r(rank, n, howmany,
                                                      ino, inembed3, istride, idist,
                                                      d_out, onembed3, ostride, odist,
                                                      kind, FFTW_ESTIMATE);
                                                      
    // calculate d2f/dx2
    memcpy(out, out_x, N1*(N2/2+1)*n3 * sizeof(fftw_complex));
    fftw_execute(bplan_3d);
    fftw_execute(plan_r2r_z_inverse);
    cout << "err d2f/dx2 = " << err_calculation(d_out, in_dfunc_xy, N1*N2*n3) << endl;

    // calculate d2f/dy2
    memcpy(out, out_y, N1*(N2/2+1)*n3 * sizeof(fftw_complex));
    fftw_execute(bplan_3d);
    fftw_execute(plan_r2r_z_inverse);
    cout << "err d2f/dy2 = " << err_calculation(d_out, in_dfunc_xy, N1*N2*n3) << endl;

    // calculate d2f/dz2
    memcpy(out, out_z, N1*(N2/2+1)*n3 * sizeof(fftw_complex));
    fftw_execute(bplan_3d);
    fftw_execute(plan_r2r_z_inverse);
    cout << "err d2f/dz2 = " << err_calculation(d_out, in_dfunc_z, N1*N2*n3) << endl;

    free(in);
    free(ino);
    free(in_dfunc_xy);
    free(in_dfunc_z);
    free(out);
    free(out_x);
    free(out_y);
    free(out_z);
    free(d_out);
    fftw_destroy_plan(plan_r2r_z_inverse);  
    fftw_destroy_plan(bplan_3d); 
    return 0;
}

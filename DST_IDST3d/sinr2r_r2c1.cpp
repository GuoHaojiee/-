#include <iostream>
#include <fftw3.h>
#include <cmath>
#include <stdlib.h>
#include <string.h>

using namespace std;

double func(double x, double y, double z)
{
	return sin(x)* cos(y)*sin(2 *M_PI*z);
}

int main() {
    int N1 = 8; 
    int N2 = 8; 
    int N3 = 8;
    int n3 = N3/2-1;
    int istride, ostride;
    int idist, odist;
    int howmany;
    int rank;
    double L_x = 2*M_PI, L_y = 2*M_PI, L_z = 1;
    double *in_ = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *in = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *ino = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *out_z = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *out_xy = (double*)fftw_malloc(sizeof(double) * N1 * N2 * N3);
    

    
    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < N2; ++j) {
            for(int k = 1; k <= n3; ++k) {
                int index = (i*N2+j)*n3+k-1; 
                in[index] = func(i*L_x/N1, j*L_y/N2, k*L_z/N3);
                in_[index] = func(i*L_x/N1, j*L_y/N2, k*L_z/N3);
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
    const fftw_r2r_kind kind[] = {FFTW_RODFT00};

    fftw_plan plan_r2r_z = fftw_plan_many_r2r(rank, n, howmany,
                                              in, inembed, istride, idist,
                                              ino, onembed, ostride, odist,
                                              kind, FFTW_ESTIMATE);

    fftw_execute(plan_r2r_z);
    fftw_destroy_plan(plan_r2r_z);

    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < N2; ++j) {
            for(int k = 0; k < n3; ++k) {
                int index = (i * N2 + j)*n3 + k;
                double mod2 = ino[index]*ino[index];
                if (mod2 > 1e-5)
                    cout << "out sin r2r " << i << " " << j << " " << k << " " << ino[index]<< endl;
            }
        }
    }
    cout << endl;
    // Forward transformation R2 -> C2 x,y
    int nn[] = {N1, N2};
    int inembed2[] =  {N1, N2};
    int onembed2[] =  {N1, N2/2+1};
    istride = n3; ostride = n3;
    idist = 1; odist = 1;
    
    fftw_complex *out = (fftw_complex*)fftw_malloc(N1*(N2/2+1)*n3 * sizeof(fftw_complex));
    fftw_plan fplan_3d = fftw_plan_many_dft_r2c(2, nn, n3,
                                                ino, inembed2, istride, idist,
                                                out, onembed2, ostride, odist,
                                                FFTW_ESTIMATE);
    fftw_execute(fplan_3d);
    fftw_destroy_plan(fplan_3d);
    

    //Normalization
    for (int i = 0; i < N1*(N2/2+1)*n3; ++i)
    {
        out[i][0] /= N1*N2*N3;
        out[i][1] /= N1*N2*N3;
    }

    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < (N2/2+1); ++j) {
            for(int k = 0; k < n3; ++k) {
                int index = (i * (N2/2+1) + j) * n3 + k;
                double mod2 = out[index][0]*out[index][0] + out[index][1]*out[index][1];
                if (mod2 > 1e-10)
                    cout << "out r2c " << i << " " << j << " " << k << " " << out[index][0] << " " << out[index][1] << endl;
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
    fftw_execute(bplan_3d);
    fftw_destroy_plan(bplan_3d);

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
                                                      out_z, onembed3, ostride, odist,
                                                      kind, FFTW_ESTIMATE);
    fftw_execute(plan_r2r_z_inverse);
    fftw_destroy_plan(plan_r2r_z_inverse);


    double err = 0.0;
    for (int i = 0; i < N1*N2*n3; ++i)
    {
        double err_ = fabs(out_z[i] - in_[i]);
        err += err_ * err_;
        //cout << out_z[i] << " " << in_[i]<< endl;
    }
    
    err = sqrt(err);
    
    cout << "err = " << err << endl;
    free(in);
    free(ino);
    free(out_z);
    free(out_xy);
    free(out);
    return 0;
}
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

double dfunc(double x, double y, double z)
{
    return  -func(x,y,z);   
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
    double *ino = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *ino_ = (double*)fftw_malloc(sizeof(double) * N1 * N2 * N3);
    double *out_z = (double*)fftw_malloc(sizeof(double) * N1 * N2 * n3);
    double *out_xy = (double*)fftw_malloc(sizeof(double) * N1 * N2 * N3);
    

    
    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < N2; ++j) {
            for(int k = 0; k < n3; ++k) {
                in[(i*N2+j)*n3+k] = func(i*L_x/N1, j*L_y/N2, k*L_z/N3);
                in_[(i*N2+j)*n3+k] = func(i*L_x/N1, j*L_y/N2, k*L_z/N3);
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
    
    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < N2; ++j) {
            for(int k = 0; k < n3; ++k) {
                //cout << i << " " << j << " " << k << " " << ino[(i*N2+j)*n3+k]<< endl;
            }
        }
    }
    
    //For the case of REDFT00, this even symmetry means that X_j = X_N-j
    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < N2; ++j) {
            ino_[(i*N2+j)*N3] = ino[(i*N2+j)*n3];
            for(int k = 1; k < n3; ++k) {
                ino_[(i*N2+j)*N3+k] = ino[(i*N2+j)*n3+k];
                ino_[(i*N2+j)*N3+N3-k] = ino[(i*N2+j)*n3+k];
            }
        }
    }
    

    // Forward transformation R2 -> C2 x,y
    int nn[] = {N1, N2};
    int inembed2[] =  {N1, N2};
    int onembed2[] =  {N1, N2/2+1};
    istride = 1; ostride = 1;
    idist = N1*N2; odist = N1*(N2/2+1);
    
    fftw_complex *out = (fftw_complex*)fftw_malloc(N1*(N2/2+1)*N3 * sizeof(fftw_complex));
    fftw_plan fplan_3d = fftw_plan_many_dft_r2c(2, nn, N3,
                                                ino_, inembed2, istride, idist,
                                                out, onembed2, ostride, odist,
                                                FFTW_ESTIMATE);
    fftw_execute(fplan_3d);
    fftw_destroy_plan(fplan_3d);
    

    //Normalization
    for (int i = 0; i < N1*(N2/2+1)*N3; ++i)
    {
        out[i][0] /= N1*N2;
        out[i][1] /= N1*N2;
    }

    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < (N2/2+1); ++j) {
            for(int k = 0; k < n3; ++k) {
                cout << i << " " << j << " " << k << " " << out[(i * N2 + j) * n3 + k][0] << " " << out[(i * N2 + j) * n3 + k][1] << endl;
            }
        }
    }
    
    // Backward transformation C2 -> R2 x,y
    inembed2[0] = N1; inembed2[1] = N2/2+1;
    onembed2[0] = N1; onembed2[1] = N2;
    istride = 1; ostride = 1;
    idist = N1*(N2/2+1); odist = N1*N2;
    fftw_plan bplan_3d = fftw_plan_many_dft_c2r(2, nn, N3,
                                                    out, inembed2, istride, idist,
                                                    out_xy, onembed2, ostride, odist,
                                                    FFTW_ESTIMATE);
    fftw_execute(bplan_3d);
    fftw_destroy_plan(bplan_3d);
    
    //Normalization
    for(int i = 0; i < N1; ++i) {
        for(int j = 0; j < N2; ++j) {
            for(int k = 0; k < n3; ++k) {
                out_xy[(i*N2+j)*N3+k] /= N3;
                in[(i*N2+j)*n3+k] = out_xy[(i*N2+j)*N3+k];
            }
        }
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
    free(ino_);
    free(out_z);
    free(out_xy);
    free(out);
    return 0;
}

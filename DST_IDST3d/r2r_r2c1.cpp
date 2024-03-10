#include <iostream>
#include <fftw3.h>
#include <cmath>
#include <stdlib.h>
#include <string.h>

using namespace std;

int main() {
    int n1 = 8; 
    int n2 = 16; 
    int n3 = 32;
    double L = 2 * M_PI;
    double *in_ = (double*)fftw_malloc(sizeof(double) * n1 * n2 * n3);
    double *in = (double*)fftw_malloc(sizeof(double) * n1 * n2 * n3);
    double *ino = (double*)fftw_malloc(sizeof(double) * n1 * n2 * n3);
    double *out_z = (double*)fftw_malloc(sizeof(double) * n1 * n2 * n3);
    fftw_complex *out = (fftw_complex*)fftw_malloc(n1*(n2/2+1)*n3 * sizeof(fftw_complex));
    for(int i = 0; i < n1; ++i) {
        for(int j = 0; j < n2; ++j) {
            for(int k = 0; k < n3; ++k) {
                in[(i * n2 + j) * n3 + k] = exp(sin(i * L / n1) * cos(j * L / n2)) * cos(2 * k * L / n3);
                in_[(i * n2 + j) * n3 + k] = exp(sin(i * L / n1) * cos(j * L / n2)) * cos(2 * k * L / n3);
            }
        }
    }

    // Forward transformation R -> R z
    int rank = 1;
    int n[] = {n3};
    int howmany = n1 * n2;
    int istride = n1 * n2, ostride = n1 * n2;
    int idist = 1, odist = 1;
    int *inembed = n, *onembed = n;
    const fftw_r2r_kind kind[] = {FFTW_REDFT00};
    fftw_plan plan_r2r_z = fftw_plan_many_r2r(rank, n, howmany,
                                              in, inembed, istride, idist,
                                              ino, onembed, ostride, odist,
                                              kind, FFTW_ESTIMATE);

    fftw_execute(plan_r2r_z);
    fftw_destroy_plan(plan_r2r_z);

    // Forward transformation R2 -> C2 x,y
    int nn[] = {n1, n2};
    int inembed2[] =  {n1, n2};
    int onembed2[] =  {n1, n2/2+1};
    istride = 1; ostride = 1;
    idist = n1*n2; odist = n1*(n2/2+1);

    fftw_plan fplan_3d = fftw_plan_many_dft_r2c(2, nn, n3,
                                                ino, inembed2, istride, idist,
                                                out, onembed2, ostride, odist,
                                                FFTW_ESTIMATE);

    fftw_execute(fplan_3d);
    fftw_destroy_plan(fplan_3d);

    //Normalization
    for (int i = 0; i < n1*(n2/2+1)*n3; ++i)
    {
        out[i][0] /= n1*n2;
        out[i][1] /= n1*n2;
    }

    // Backward transformation C2 -> R2 x,y
    inembed2[0] = n1; inembed2[1] = n2/2+1;
    onembed2[0] = n1; onembed2[1] = n2;
    istride = 1; ostride = 1;
    idist = n1*(n2/2+1); odist = n1*n2;
    fftw_plan bplan_3d = fftw_plan_many_dft_c2r(2, nn, n3,
                                                    out, inembed2, istride, idist,
                                                    in, onembed2, ostride, odist,
                                                    FFTW_ESTIMATE);
    fftw_execute(bplan_3d);
    fftw_destroy_plan(bplan_3d);
    
    //Normalization
    for (int i = 0; i < n1*n2*n3; ++i)
    {
        in[i] /= 2*(n3-1);
    }

    // Backward transformation R -> R z
    rank = 1;
    n[0] = n3;
    howmany = n1 * n2;
    istride = n1 * n2; ostride = n1 * n2;
    idist = 1; odist = 1;
    int *inembed3 = n, *onembed3 = n;
    fftw_r2r_kind kind_z_inverse[] = {FFTW_REDFT00};
    fftw_plan plan_r2r_z_inverse = fftw_plan_many_r2r(rank, n, howmany,
                                                      in, inembed3, istride, idist,
                                                      out_z, onembed3, ostride, odist,
                                                      kind, FFTW_ESTIMATE);
    fftw_execute(plan_r2r_z_inverse);
    fftw_destroy_plan(plan_r2r_z_inverse);


    double err = 0.0;
    for (int i = 0; i < n1*n2*n3; ++i)
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
    free(out);
    return 0;
}
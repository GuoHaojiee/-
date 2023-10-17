#include<iostream>
#include<stdlib.h>
#include<omp.h>

using namespace std;
int main(){
    omp_set_num_threads(6);
    int n = 100000; 
    double a = 2.0; 
    double b = 3.0; 
    double* V = new double[n];
    double* U = new double[n];
    double* W_para = new double[n];
    double* W_seri = new double[n];

    double start1,end1,start2, end2, time_seri, time_para;
    for (int i = 0; i < n; i++) { //инициализация
        V[i] = (double)rand() / RAND_MAX; 
        U[i] = (double)rand() / RAND_MAX;
    }
    start1 = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        W_seri[i] = a * V[i] + b * U[i];
        }
    end1 = omp_get_wtime();
    time_seri = end1 - start1;

    start2 = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        W_para[i] = a * V[i] + b * U[i];
    }
    end2 = omp_get_wtime();
    time_para = end2 - start2;
    cout << "time serial = " << time_seri * 1000 << "ms" << endl;
    cout << "time parallel = " << time_para * 1000 << "ms" << endl;

    bool is_same = true; 
    for(int i = 0; i < n; i++) {
        if (W_para[i] != W_seri[i])
            is_same = false;
    }
    if (is_same) cout << "Results are same" << endl;
    else cout << "Results are different" << endl; 
    delete [] W_seri;
    delete [] W_para; 
    delete [] U;
    delete [] V;
    return 0;
}


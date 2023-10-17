#include<iostream>
#include<stdlib.h>
#include<omp.h>

using namespace std;
int main(){
    omp_set_num_threads(6);
    int n = 10000; 
    double a = 2.0; 
    double b = 3.0; 
    double** matrix = new double*[n]; // matrix[n][n]
    for (int i = 0; i <= n; ++i) {
        matrix[i] = new double[n];
    }
    double* vector = new double[n];
    double* res1 = new double[n];
    double* res2 = new double[n];
    double start1,end1,start2, end2, time_seri, time_para;
    for (int i = 0; i < n; i++) { //инициализация
        for( int j = 0; j < n; j++)
            matrix[i][j] = (double)rand() / RAND_MAX;
        vector[i] = (double)rand() / RAND_MAX;
    }
    start1 = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        double result = 0.0;
        for (int j = 0; j < n ; j++){
            result += matrix[i][j] * vector[j];
        }
        res1[i] = result; 
    }
    end1 = omp_get_wtime();
    time_seri = end1 - start1;

    start2 = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        double result = 0.0;
        for (int j = 0; j < n; j++) {
            result += matrix[i][j] * vector[j];
        }
        res2[i] = result;
    }

    end2 = omp_get_wtime();
    time_para = end2 - start2;
    cout << "time serial = " << time_seri * 1000 << "ms" << endl;
    cout << "time parallel = " << time_para * 1000 << "ms" << endl;
   
    bool is_same = true; 
    for(int i = 0; i < n; i++) {
        if ( res1[i] != res2[i])
            is_same = false;
    }
    if (is_same) cout << "Results are same" << endl;
    else cout << "Results are different" << endl; 

    delete [] res2;
    delete [] res1;
    delete [] vector;
    for(int i = 0; i < n; i++){
        delete[] matrix[i]; 
    }
    delete [] matrix;
    return 0;
}


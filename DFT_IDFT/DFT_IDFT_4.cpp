#include <iostream>
#include <cmath>
#include <fftw3.h>

using namespace std;

double u_f(double t, double x){ //u(t, x) = (t^2 + 1)sin(2*pi*x)
    return (t*t + 1) * sin(2* M_PI *x);
}

double fi(double x){ // fi(x) =  u(0, x)
    return sin(2* M_PI *x);
}

double pi_1(double t){ // u(t, 0) = pi_1(t) = 0
    return 0;
}

double pi_2(double t){ // u(t, 1) = pi_2(t) = (t^2 + 1)sin（2 * pi）
    return (t*t + 1)* sin(2* M_PI);  
}

double f_(double t, double x){
    return 2*t*sin(2 * M_PI * x) + 4* M_PI * M_PI *(t*t + 1) * sin(2 * M_PI * x); 
}

bool is_good_h_tau(double h, double tau){
    if ( tau < ((h*h)/2)) return true;
    else return false;
}

int main(){
    cout.setf(ios::fixed,ios::floatfield);
    cout.precision(10);
    double L = 1;
    double T = 1;
    int M, N;
    cout << "Please input M , N : " << endl;
    cin >> M >> N;
    double h = L / M;
    double tau = T / N;
    if ((is_good_h_tau(h,tau)) == false) {
        cout << "M, N are not good" << endl;
        return 0; 
    }

    //Выделение памяти:
    double** u = (double**) fftw_malloc(sizeof(double*) * (N+1));
    double** f = (double**) fftw_malloc(sizeof(double*) * (N+1));
    fftw_complex** u_k = (fftw_complex**) fftw_malloc(sizeof(fftw_complex*) * (N+1));
    fftw_complex** f_k = (fftw_complex**) fftw_malloc(sizeof(fftw_complex*) * (N+1));
    for (int i = 0; i <= N; ++i) {
        u[i] = (double*) fftw_malloc(sizeof(double) * (M+1));
        f[i] = (double*) fftw_malloc(sizeof(double) * (M+1));
        u_k[i] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (M+1)/2+1);
        f_k[i] = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (M+1)/2+1);
    }

    for (int i = 0; i <= M; ++i) { 
        u[0][i] = fi(i * h);    //Начальные условия
    }
    
    for(int n = 0; n <= N; n++){
        for (int i = 0; i <= M; ++i) { 
            f[n][i] = f_(n * tau, i * h);        
        }
    }
    
    //Преобразование Фурье
    fftw_plan p1 = fftw_plan_dft_r2c_1d(M+1, u[0] , u_k[0], FFTW_ESTIMATE);
    fftw_execute(p1);

    for (int k = 0; k < (M+1)/2+1; k++) {
        cout << u_k[0][k][0] << " " << u_k[0][k][1] << endl;
        u_k[0][k][0] /= (M+1)/2 +1;
        u_k[0][k][1] /= (M+1)/2 +1;
    }

    double alpha = 2 * M_PI/L;
    for(int n = 1; n <= N; n++){
        fftw_plan p2 = fftw_plan_dft_r2c_1d(M+1, f[n-1] , f_k[n-1] , FFTW_ESTIMATE);
        fftw_execute(p2);
        fftw_destroy_plan(p2);
        for(int k = 0; k < (M+1)/2+1; k++){
            u_k[n][k][0] = u_k[n-1][k][0] - tau * alpha*alpha*k*k*u_k[n-1][k][0] + tau*f_k[n-1][k][0]/(M+1); 
            u_k[n][k][1] = u_k[n-1][k][1] - tau * alpha*alpha*k*k*u_k[n-1][k][1] + tau*f_k[n-1][k][1]/(M+1);   
        }
        fftw_plan p3 = fftw_plan_dft_c2r_1d(M+1, u_k[n] , u[n], FFTW_ESTIMATE);
        fftw_execute(p3);
        fftw_destroy_plan(p3);
    }
    for (int n = 0; n <= N; n++) {
        u[n][0] = pi_1(n * tau);
        u[n][M] = pi_2(n * tau);
    }

    //Вычисление ошибки:
    double error1, error2;
    double sum = 0;
    for(int i = 0; i <= M; i++){
        sum += pow((u_f(1, i * h) - u[N][i]), 2);    
        //cout << u_f( tau * (1), i * h) << " "<< u[1][i] << endl;
    }
    error1 = sqrt(sum * h);
    double max = u_f(1, 0 * h) - u[N][0] ;
    double value;
    for(int i = 0; i <= M; i++){
        value = abs(u_f(1, i * h) - u[N][i]);
        if (value > max)
            max = value;    
    }
    cout << error1 << endl;
    cout << max << endl;
    
    //Освобождение памяти:
    for (int i = 0; i <= N; ++i) {
        fftw_free(u[i]);
        fftw_free(u_k[i]);
        fftw_free(f[i]);
        fftw_free(f_k[i]);
    }
   // fftw_destroy_plan(p1);
    fftw_free(u);
    fftw_free(u_k);
    fftw_free(f);
    fftw_free(f_k);
    return 0;
}

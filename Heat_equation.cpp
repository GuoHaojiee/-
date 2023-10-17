#include <iostream>
#include <cmath>

using namespace std;

double u_f(double t, double x){ //u(t, x) = (t^2 + 1)sin2x
    return (t*t + 1) * sin(2*x);
}

double fi(double x){ // fi(x) =  u(0, x)
    return sin(2*x);
}

double pi_1(double t){ // u(t, 0) = pi_1(t) = 0
    return 0;
}

double pi_2(double t){ // u(t, 1) = pi_2(t) = (t^2 + 1)sin2 
    return (t*t + 1)* sin(2);  
}

double f(double t, double x){
    return 2*t*sin(2*x) + 4*(t*t + 1) * sin(2*x); 
}

bool is_good_h_tal(double h, double tal){
    if ( tal < ((h*h)/2)) return true;
    else return false;
}

int main(){
    cout.setf(ios::fixed,ios::floatfield);
    cout.precision(10);
    double L = 1;
    double T = 1;
    double h, tal; 
    int M, N;
    cout << "Please input M , N : " << endl;
    cin >> M >> N;
    h = L / M; tal = T / N;
    if ((is_good_h_tal(h,tal)) == false) {
        cout << "M, N are not good" << endl;
        return 0; 
    }
    double error1, error2;
    double a_tal_h  = tal / (h*h);
    double sum = 0;
    
    double** u = new double*[N+1]; // u[N][M]
    for (int i = 0; i <= N; ++i) {
        u[i] = new double[M+1];
    }

    for (int i = 0; i <= M; ++i) { 
        u[0][i] = fi(i * h);
    }

    for (int n = 0; n <= N; ++n) {
        u[n][0] = pi_1(n * tal);
        u[n][M] = pi_2(n * tal);
    }

    for(int n = 1; n <= N; n++){
        for(int i = 1; i <= M-1; i++){
            u[n][i] = u[n-1][i] + a_tal_h*( u[n-1][i+1] - 2*u[n-1][i] + u[n-1][i-1]) + tal*f((n-1) * tal, i * h);   
        }
    }

    for(int i = 0; i <= M; i++){
        sum += pow((u_f(1, i * h) - u[N][i]), 2);    
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
    for (int i = 0; i <= N; ++i) {
        delete[] u[i];
    }
    delete[] u;
    return 0;
}

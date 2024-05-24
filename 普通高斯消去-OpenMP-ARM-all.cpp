#include<iostream>
#include<windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
using namespace std;
#define N 2000
#define NUM_THREADS 7
float** A = NULL;  //?????????

long long head, tail, freq;


void A_init() {     //��??????????????
    A = new float* [N];
    for (int i = 0; i < N; i++) {
        A[i] = new float[N];
    }
    for (int i = 0; i < N; i++) {
        A[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            A[i][j] = rand() % 1000;
        }

    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] += A[k][j];
                A[i][j] = (int)A[i][j] % 1000;
            }
        }
    }
}


void deleteA() {
    for (int i = 0; i < N; i++) {
        delete[] A[i];
    }
    delete A;
}


void print() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }
}

void LU() {    //????????
    int i, j, k;
    for (k = 0; k < N; k++) {
        for (j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;
        for ( i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void LU_omp() {    //
    int i =0 , j = 0, k = 0;
    float tmp = 0;
    #pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++) {
        #pragma omp single
        { tmp = A[k][k];
        for (j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / tmp;
        }
        A[k][k] = 1.0;
        }

        #pragma omp for
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void LU_omp_guided() {    //
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++) {
#pragma omp single
        { tmp = A[k][k];
        for (j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / tmp;
        }
        }
        A[k][k] = 1.0;
#pragma omp for schedule(guided)
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}

void LU_omp_dynamic() {    //
    int i = 0, j = 0, k = 0;
    float tmp = 0;
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j,k,tmp)
    for (k = 0; k < N; k++) {
#pragma omp single
        { tmp = A[k][k];
        for (j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / tmp;
        }
        }
        A[k][k] = 1.0;
#pragma omp for schedule(dynamic)
        for (i = k + 1; i < N; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}


void cal(void(*func)()) {
    A_init();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    func();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

}


int main() {
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    cal(LU);
    cout << "????????��????" << (tail - head) * 1000 / freq << "ms" << endl;
    deleteA();

    cal(LU_omp);
    cout << "?????OpenMp(_static)?????" << (tail - head) * 1000 / freq << "ms" << endl;
    deleteA();

    cal(LU_omp_guided);
    cout << "?????OpenMp_guided?????" << (tail - head) * 1000 / freq << "ms" << endl;
    deleteA();

    cal(LU_omp_dynamic);
    cout << "?????OpenMp_dynamic?????" << (tail - head) * 1000 / freq << "ms" << endl;
    deleteA();


}


#include <iostream>
#include <Windows.h>
#include <immintrin.h>
using namespace std;
const int _NUM = 500;
float matrix[_NUM][_NUM];

void Generate(float A[][_NUM], int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < i; j++)
			A[i][j] = 0.0;
		A[i][i] = 1.0;
		for (int j = i + 1; j < size; j++)
			A[i][j] = rand() % 10;
	}
	for (int k = 0; k < size; k++)
		for (int i = k + 1; i < size; i++)
			for (int j = 0; j < size; j++)
				A[i][j] += A[k][j];
}
void Gauss_AVX512(float A[][_NUM], int NUM) {
	for (int k = 0; k < NUM; k++) {
		__m512 v1 = _mm512_set1_ps(A[k][k]);
		int j;
		for (j = k + 1; j + 16 <= NUM; j += 16) {
			//A[k][j] /= A[k][k];
			__m512 va = _mm512_loadu_ps(A[k] + j);
			va = _mm512_div_ps(va, v1);
			_mm512_storeu_ps(A[k] + j, va);
		}
		for (; j < NUM; j++)
			A[k][j] /= A[k][k];
		A[k][k] = 1.0;
		for (int i = k + 1; i < NUM; i++) {
			__m512 vaik = _mm512_set1_ps(A[i][k]);
			for (j = k + 1; j + 16 <= NUM; j += 16) {
				//A[i][j] = A[i][j] - A[i][k] * A[k][j];
				__m512 vaij = _mm512_loadu_ps(A[i] + j);
				__m512 vakj = _mm512_loadu_ps(A[k] + j);
				__m512 vx = _mm512_mul_ps(vaik, vakj);
				vaij = _mm512_sub_ps(vaij, vx);
				_mm512_storeu_ps(A[i] + j, vaij);
			}
			for (; j < NUM; j++)
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			A[i][k] = 0;
		}
	}
}

int main() {

	Generate(matrix, _NUM);

	long long head, tail, freq;
	double sum_time = 0.0;
	int ecx = 1;
	while (ecx > 0) {
		Generate(matrix, _NUM);
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		Gauss_AVX512(matrix, _NUM);
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		sum_time += (tail - head) * 1000.0 / freq;
		ecx--;
	}
	cout <<sum_time / double(1.0) <<  endl;
	return 0;
}

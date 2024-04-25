#include <iostream>
#include <stdio.h>
#include <windows.h>
#include <stdlib.h>
#include <immintrin.h>
using namespace std;
long long head, tail, freq;        // timers
const int n = 2048;
int t = 13;
int T = 1;
int test[20] = { 10,30,50,100,150,200,300,400,500,700,1000,1500,2000 };
float a[n][n];
float result[n][n]{};
void m_reset(int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
            a[i][j] = 0;
        a[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            a[i][j] = rand();
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                a[i][j] += a[k][j];


}
void start()
{
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
}
void timestop()
{
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);	// end time


}
void pingfan(int N)
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
        {
            a[k][j] = a[k][j] / a[k][k];

        }
        m[k][k] = 0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
            {
                a[i][j] -= a[i][k] * a[k][j];
            }
            a[i][k] = 0;
        }

    }

}
int main()
{
    for (int j = 0; j < T; j++)
        {
        for (int i = 0; i < 13; i++)
        {
            m_reset(test[i]);
            QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            pingfan(test[i]);
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            cout << ((tail - head) * 1000.0 / freq)/1000 << endl;
        }
    }
}

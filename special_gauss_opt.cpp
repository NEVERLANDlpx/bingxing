#include<iostream>
#include<fstream>
#include<cmath>
#include<string>
#include<chrono>
#include <xmmintrin.h> 
#include <emmintrin.h> 
#include <pmmintrin.h> 
#include <tmmintrin.h> 
#include <smmintrin.h> 
#include <nmmintrin.h>
#include <immintrin.h> 
using namespace std;
using namespace chrono;
#define N 3799 //矩阵列数
#define C2 1953 //被消元行行数
int s[N][N] = { 0 }, a[C2][N] = { 0 }, reach[C2] = { 0 };

void initialize()
{
	for (int i = 0; i < N; i++)
	{
		if (i < C2)
			reach[i] = 0;
		for (int j = 0; j < N; j++)
		{
			if (i < C2)
				a[i][j] = 0;
			s[i][j] = 0;
		}
	}
	int t1, t2, i;
	fstream fin1("a1.txt");
	fin1 >> t1;
	i = N - t1 - 1;
	s[i][i] = 1;
	while (!fin1.eof())
	{
		fin1 >> t2;
		if (t2 > t1)
		{
			i = N - t2 - 1;
			//s[i][i] = 1;
			s[i][i] |= 1;
			t1 = t2;
		}
		else
		{
			s[i][N - t2 - 1] |= 1;
			t1 = t2;
		}
	}
	fin1.close();

	fstream fin2("a2.txt");
	i = 0;
	fin2 >> t1;
	a[i][N - t1 - 1] = 1;
	while (!fin2.eof())
	{
		fin2 >> t2;
		if (t2 > t1)
		{
			a[++i][N - t2 - 1] |= 1;
			t1 = t2;
		}
		else
		{
			a[i][N - t2 - 1] |= 1;
			t1 = t2;
		}
	}
	fin2.close();
}



void print()
{
	for (int i = 0; i < C2; i++)
	{
		for (int j = 0; j < N; j++)
			cout << a[i][j] << " ";
		cout << endl;
	}
	cout << endl;
}

void Plain(int s[][N], int a[][N])
{
	int i, j, k, m;
	for (i = 0; i + 5 < N; i += 5)//每次导入五行消元子
	{
		for (j = 0; j < C2; j++)
		{
			if (reach[j] == 0)
				for (k = 0; k < i + 5; k++)
				{
					if (a[j][k] == 1 && s[k][k] == 1)
					{
						for (m = k; m < N; m++)
						{
							a[j][m] = a[j][m] ^ s[k][m];
						}
					}
					else if (a[j][k] == 1 && s[k][k] == 0)
					{
						for (m = k; m < N; m++)
						{
							s[k][m] = a[j][m];
						}
						reach[j] = 1;
						break;
					}
				}
		}
	}
	for (i; i < N; i++)//剩下的部分
	{
		for (j = 0; j < C2; j++)
		{
			if (reach[j] == 0)
				for (k = 0; k < N; k++)
				{
					if (a[j][k] == 1 && s[k][k] == 1)
					{
						for (m = k; m < N; m++)
						{
							a[j][m] = a[j][m] ^ s[k][m];
						}
					}
					else if (a[j][k] == 1 && s[k][k] == 0)
					{
						for (m = k; m < N; m++)
						{
							s[k][m] = a[j][m];
						}
						reach[j] = 1;
						break;
					}
				}
		}
	}
}

void SSE(int s[][N], int a[][N])
{
	int i, j, k, m;
	__m128i a1, a2;
	for (i = 0; i + 5 < N; i += 5)//每次导入五行消元子
	{
		for (j = 0; j < C2; j++)
		{
			if (reach[j] == 0)
				for (k = 0; k < i + 5; k++)
				{
					if (a[j][k] == 1 && s[k][k] == 1)
					{
						for (m = k; m + 4 < N; m += 4)
						{
							//a[j][m] = a[j][m] ^ s[k][m];
							a1 = _mm_load_si128((__m128i*)s[k] + m);
							a2 = _mm_load_si128((__m128i*)a[j] + m);
							_mm_store_si128((__m128i*)a[j] + m, _mm_xor_si128(a1, a2));
						}
						for (m; m < N; m++)//剩余部分
						{
							a[j][m] = a[j][m] ^ s[k][m];
						}
					}
					else if (a[j][k] == 1 && s[k][k] == 0)
					{
						for (m = k; m < N; m++)
						{
							s[k][m] = a[j][m];
						}
						reach[j] = 1;
						break;
					}
				}
		}
	}
	for (i; i < N; i++)//剩下的部分
	{
		for (j = 0; j < C2; j++)
		{
			if (reach[j] == 0)
				for (k = 0; k < N; k++)
				{
					if (a[j][k] == 1 && s[k][k] == 1)
					{
						for (m = k; m < N; m++)
						{
							a[j][m] = a[j][m] ^ s[k][m];
						}
					}
					else if (a[j][k] == 1 && s[k][k] == 0)
					{
						for (m = k; m < N; m++)
						{
							s[k][m] = a[j][m];
						}
						reach[j] = 1;
						break;
					}
				}
		}
	}
}

void AVX(int s[][N], int a[][N])
{
	int i, j, k, m;
	__m256i a1, a2;
	for (i = 0; i + 5 < N; i += 5)//每次导入五行消元子
	{
		for (j = 0; j < C2; j++)
		{
			if (reach[j] == 0)
				for (k = 0; k < i + 5; k++)
				{
					if (a[j][k] == 1 && s[k][k] == 1)
					{
						for (m = k; m + 8 < N; m += 8)
						{
							//a[j][m] = a[j][m] ^ s[k][m];
							a1 = _mm256_load_si256((__m256i*)s[k] + m);
							a2 = _mm256_load_si256((__m256i*)a[j] + m);
							_mm256_store_si256((__m256i*)a[j] + m, _mm256_xor_si256(a1, a2));
						}
						for (m; m < N; m++)//剩余部分
						{
							a[j][m] = a[j][m] ^ s[k][m];
						}
					}
					else if (a[j][k] == 1 && s[k][k] == 0)
					{
						for (m = k; m < N; m++)
						{
							s[k][m] = a[j][m];
						}
						reach[j] = 1;
						break;
					}
				}
		}
	}
	for (i; i < N; i++)//剩下的部分
	{
		for (j = 0; j < C2; j++)
		{
			if (reach[j] == 0)
				for (k = 0; k < N; k++)
				{
					if (a[j][k] == 1 && s[k][k] == 1)
					{
						for (m = k; m < N; m++)
						{
							a[j][m] = a[j][m] ^ s[k][m];
						}
					}
					else if (a[j][k] == 1 && s[k][k] == 0)
					{
						for (m = k; m < N; m++)
						{
							s[k][m] = a[j][m];
						}
						reach[j] = 1;
						break;
					}
				}
		}
	}
}



void store()
{
	ofstream fout("out.txt");
	for (int i = 0; i < C2; i++)
	{
		bool flag = false;
		for (int j = 0; j < N; j++)
		{
			if (a[i][j] == 1)
			{
				fout << N - j - 1 << " ";
				flag = true;
			}
		}
		if (flag)
			fout << endl;
	}
}



int main()
{
	initialize();
	typedef std::chrono::high_resolution_clock Clock;
	auto t1 = Clock::now();
	Plain(s, a);
	auto t2 = Clock::now();
	store();
	cout << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 << endl;

	initialize();
	auto t3 = Clock::now();
	SSE(s, a);
	auto t4 = Clock::now();
	store();
	cout << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 1e+6 << endl;

	initialize();
	auto t5 = Clock::now();
	AVX(s, a);
	auto t6 = Clock::now();
	store();
	cout << std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t5).count() / 1e+6 << endl;
}

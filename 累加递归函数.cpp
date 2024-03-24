#include <iostream>
#include <windows.h>
#include <cstdlib>
using namespace std;
int n;
float sum,a[1500],tsum;
int c[9]={10,30,50,70,100,300,500,700,1000};
void init()
{
    for(int i=0;i<n;i++) a[i]=i;
}
void recursion(int n)
{
     if (n == 1)
     return;
     else
     {
     for (int i = 0; i < n / 2; i++)
     a[i ] += a[n - i -1];
     n = n / 2;
     recursion(n);
     }
}
int main()
{
    for(int w=0;w<9;w++){
	n=c[w];
    init();
    tsum=0.0;
    for(int k=1;k<=50;k++)
    {
    long long head, tail , freq ;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq ) ;
    //start time
    QueryPerformanceCounter((LARGE_INTEGER *)&head) ;
/* 递归：
1. 将给定元素两两相加，得到n/2个中间结果;
2. 将上一步得到的中间结果两两相加，得到n/4个中间结果;
3. 依此类推，log(n)个步骤后得到一个值即为最终结果。
*/
  recursion(n);
   // end time
   QueryPerformanceCounter((LARGE_INTEGER *)&tail ) ;
   
   tsum+=(tail -head)* 1000.0 / freq;
   }
   cout<<"n="<<c[w]<<",time="<<tsum/50.0<<"ms"<<endl;
   }
    return 0;
}
/*
n=10,time=8.6e-005ms
n=30,time=0.000168ms
n=50,time=0.00029ms
n=70,time=0.000404ms
n=100,time=0.00082ms
n=300,time=0.001606ms
n=500,time=0.002076ms
n=700,time=0.003856ms
n=1000,time=0.00407ms
*/

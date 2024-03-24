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
   // 实现方式2：二重循环
    for (int m = n; m > 1; m /= 2) // log(n)个步骤
      for (int i = 0; i < m / 2; i++)
        a[i ] = a[i * 2] + a[i * 2 + 1]; // 相邻元素相加连续存储到数组最前面
   // a[0]为最终结果

   // end time
   QueryPerformanceCounter((LARGE_INTEGER *)&tail ) ;
   
   tsum+=(tail -head)* 1000.0 / freq;
   }
   cout<<"n="<<c[w]<<",time="<<tsum/50.0<<"ms"<<endl;
   }
    return 0;
}
/*
n=10,time=5.6e-005ms
n=30,time=0.000114ms
n=50,time=0.000186ms
n=70,time=0.000272ms
n=100,time=0.00039ms
n=300,time=0.00139ms
n=500,time=0.001996ms
n=700,time=0.00274ms
n=1000,time=0.003478ms
*/

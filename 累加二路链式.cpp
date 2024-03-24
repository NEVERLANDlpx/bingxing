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
    float sum1 = 0,sum2 = 0,sum;
    for (int i = 0;i < n; i += 2) {
    sum1 += a[i];
    sum2 += a[i + 1];
    }
    sum = sum1 + sum2;

// end time
   QueryPerformanceCounter((LARGE_INTEGER *)&tail ) ;
   
   tsum+=(tail -head)* 1000.0 / freq;
   }
   cout<<"n="<<c[w]<<",time="<<tsum/50.0<<"ms"<<endl;
   }
    return 0;
}
/*
n=10,time=3.4e-005ms
n=30,time=8.4e-005ms
n=50,time=0.00013ms
n=70,time=0.000172ms
n=100,time=0.000242ms
n=300,time=0.000572ms
n=500,time=0.000912ms
n=700,time=0.001328ms
n=1000,time=0.001798ms
*/

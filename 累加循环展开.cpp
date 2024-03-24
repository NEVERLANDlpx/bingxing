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
    double sum=0.0;  
    const int unrollFactor=4; // 假设展开4次  
    for (int i=0;i<n;i+=unrollFactor) 
	{  sum+=a[i]+a[i+1]+a[i+2]+a[i+3];  }  
// 处理剩余的元素（如果有的话）  
    for (int i=n-(n%unrollFactor);i<n;++i) 
	{   sum += a[i];  }
// end time
   QueryPerformanceCounter((LARGE_INTEGER *)&tail ) ;
   
   tsum+=(tail -head)* 1000.0 / freq;
   }
   cout<<"n="<<c[w]<<",time="<<tsum/50.0<<"ms"<<endl;
   }
    return 0;
}
/*
n=10,time=2.6e-005ms
n=30,time=3.4e-005ms
n=50,time=5e-005ms
n=70,time=5.6e-005ms
n=100,time=7.4e-005ms
n=300,time=0.0002ms
n=500,time=0.000288ms
n=700,time=0.000402ms
n=1000,time=0.00057ms
*/

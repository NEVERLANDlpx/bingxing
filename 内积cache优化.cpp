#include <iostream>
#include <windows.h>
#include <cstdlib>
using namespace std;
int n;
float sum[1500],b[1500][1500],a[1500],tsum;
int c[9]={10,30,50,70,100,300,500,700,1000};
void init()
{

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++)
        {
            b[j][i]=j+i;
        }
        a[i]=i;
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

    for(int i = 0; i < n; i++)
	  sum[i] = 0.0;
    for(int j = 0; j < n; j++)
     for(int i = 0; i < n; i++)
     sum[i] += b[j][i] * a[j ];
// end time
   QueryPerformanceCounter((LARGE_INTEGER *)&tail ) ;
   
   tsum+=(tail -head)* 1000.0 / freq;
   }
   cout<<"n="<<c[w]<<",time="<<tsum/50.0<<"ms"<<endl;
   }
    return 0;
}
/*
n=10,time=0.000748ms
n=30,time=0.007052ms
n=50,time=0.014202ms
n=70,time=0.027462ms
n=100,time=0.042636ms
n=300,time=0.287156ms
n=500,time=0.546156ms
n=700,time=1.03572ms
n=1000,time=2.11419ms
*/

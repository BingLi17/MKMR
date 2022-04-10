#include "mex.h"
#include <math.h>
    
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
int k,k1,i,j,n;
double *K, *z, *alpha;

K = mxGetPr(prhs[0]); 
alpha = mxGetPr(prhs[1]); 
n =  mxGetM(prhs[1]);

plhs[0]=mxCreateDoubleMatrix(n,1,0); 
z= mxGetPr(plhs[0]); 
for (j=0;j<=n-1;j++) z[j]=0;
k=0;
k1=0;
for (j=0;j<=n-1;j++)
	{	
	for (i=0;i<=j;i++)
		{
		z[i]+=K[k]*alpha[j];
		if (i!=j) z[j]+=K[k]*alpha[i];
		k++;		
		}
	k1 += n;
	}
}



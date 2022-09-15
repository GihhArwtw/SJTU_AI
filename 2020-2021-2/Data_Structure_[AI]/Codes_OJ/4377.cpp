#include <cstdio>
#define MOD 1000000007
using namespace std;

int main()
{
	/*  Catalan Number. */
	/* f(n) = f(0)*f(n-1) (1 as root) + f(1)*f(n-2) (2 as root) + ... + f(n-1)*f(0) (n as root) */
	int n;
	scanf("%d",&n);
	long long* f = new long long[n+5];
	f[0] = 1; f[1] = 1;
	for (int i=2; i<=n; i++)
	{
		f[i] = 0;
		for (int j=0; j<i; j++)	f[i] += (f[j]*f[i-j-1])%MOD;
		f[i] = f[i] % MOD;
	}
	printf("%d",f[n]);
	delete []f;
	
	return 0;
} 

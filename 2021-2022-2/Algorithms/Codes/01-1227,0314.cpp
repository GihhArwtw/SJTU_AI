#include<cstdio>
#include<complex>
using namespace std;

typedef complex<double> cpx;
const double pi = 3.1415926535898;
const int MAXN = 800010;

cpx c[MAXN];
cpx h[MAXN];

void FFT(cpx* a, cpx* f, int l, int n, int pow)
{
	if (n==1)
	{
		f[l] = a[l];
		return;
	}
	
	int tmp;
	for (int i=0; i<n/2; i++)
	{
		c[l+i] = a[l+i*2];           // Even Part. Move to the front part.
		c[l+n/2+i] = a[l+i*2+1];     // Odd  Part. Move to the rear part. 
	}
	for (int i=l; i<l+n; i++)	a[i] = c[i];
	FFT(a,f,l,n/2,pow);
	FFT(a,f,l+n/2,n/2,pow);
	
	for (int i=0; i<n/2; i++)        // Restore coefficients.
	{
		c[l+i*2] = a[l+i];
		c[l+i*2+1] = a[l+n/2+i];
	}
	
	cpx omega(0,1);                  // omega = i
	omega = exp(pow*2*pi/n*omega);   // omega = exp(pow*2*pi*i/n)
	cpx w(1,0);
	for (int i=l; i<l+n/2; i++)
	{
		h[i] = f[i] + w * f[n/2+i];    // f[i]:A_even(w^(2i));  f[n/2+i]:A_odd(w^(2i))
		w = w * omega;
	}
	for (int i=l+n/2; i<l+n; i++)
	{
		h[i] = f[i-n/2] + w * f[i];    // f[i]:A_even(w^(2i));  f[n/2+i]:A_odd(w^(2i))
		w = w * omega;
	}
		
	for (int i=l; i<l+n;i++)
	{
		a[i] = c[i];
		f[i] = h[i];
	}
}

int main()
{
	int n,m;
	scanf("%d %d",&n,&m);
	
	int bit=1;
	while (bit<=n+m)	bit <<= 1;
	
	
	int x;
	cpx a[MAXN], b[MAXN];
	for (int i=0; i<=n; i++)	
	{
		scanf("%d",&x);
		a[i] = cpx(x,0);
	}
	for (int i=n+1; i<bit; i++) a[i] = cpx(0,0);
	for (int i=0; i<=m; i++)
	{
		scanf("%d",&x);
		b[i] = cpx(x,0);
	}
	for (int i=m+1; i<bit; i++) b[i] = cpx(0,0);
	
	cpx f[MAXN], g[MAXN];
	FFT(a,f,0,bit,1);
	FFT(b,g,0,bit,1);
	
	for (int i=0; i<bit; i++)	f[i] = f[i]*g[i];
	FFT(f,a,0,bit,-1);
	
	int N = n+m+1;						              // (n+1)+(m+1)-1=n+m+1
	
	for (int i=0; i<N; i++)
		printf("%d ", int(a[i].real()/double(bit)+0.5));
	return 0;
}
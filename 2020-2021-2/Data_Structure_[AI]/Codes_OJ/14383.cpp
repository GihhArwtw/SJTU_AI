#include <cstdio>
using namespace std;
#define lint long long

int maintain(const lint tmpa, const lint tmpb, lint* fa, lint* fb, const int size)
{
	int hole = 1, child;
	while (hole*2<=size)
		{
			child = hole*2;
			if (hole*2+1<=size && fa[child]*fb[child+1]<fa[child+1]*fb[child]) child++;
			if (fa[child]*tmpb>tmpa*fb[child])
			{
				fa[hole] = fa[child];
				fb[hole] = fb[child];
			}
			else break;
			hole = child;
		}
	fa[hole] = tmpa;
	fb[hole] = tmpb;
} 

int main()
{
	int n, k;
	
	scanf("%d %d",&n,&k);
	int size = n-1;
	lint* fa = new lint[n+10]; // Priority Queue
	lint* fb = new lint[n+10];
	
	for (int i=2; i<=n; i++)  { fa[n-i+1] = i-1; fb[n-i+1] = i;  }
		
	int maxa, maxb, tmpa, tmpb, x, y, r;
	for (int i=1; i<k; i++)
	{
		tmpa = fa[1]; tmpb = fb[1]; y = 2;
		while (y>1)
		{
			x = --tmpa; y = fb[1]; r = x % y;
			if (x<1) break;
			while (r) { x = y; y = r; r = x % y;	}
		}
		if (x<1)
		{
		//	tmpa = 0;
		//	tmpb = 1;
			if (size==1) break;
			maintain(fa[size],fb[size],fa,fb,size-1);
			size--;
			continue;
		}
		maintain(tmpa,tmpb,fa,fb,size);
		if (tmpa==0) size--;
	}
	printf("%d/%d",fa[1],fb[1]);
	delete []fa;
	delete []fb;

	return 0;
}

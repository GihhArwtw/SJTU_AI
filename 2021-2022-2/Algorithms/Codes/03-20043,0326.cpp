#include <cstdio>
using namespace std;

int main()
{
	int n,w,p;
	scanf("%d %d %d",&n,&w,&p);
	if (n==1)
	{
		printf("0");
		return 0;
	}
	
	int* a = new int[n+10];
	int x, tmp;
	for (int i=1; i<=n; i++)
	{
		scanf("%d",&tmp);
		a[i] = tmp;	x = i;
		while (x>1 && tmp<a[x>>1])
		{
			a[x] = a[x>>1];
			x >>= 1;
		}
		a[x] = tmp;
	}
	
	// Now we use "a" as a heap.
	int child;
	long long total=0;
	for (int i=1; i<n; i++)
	{
		tmp = 0;
		for (int j=0; j<2; j++)            // merge the two smallest.
		{
			tmp += a[1];                   // pop min.
			a[1] = a[n-i-j+1];
			x = 1;
			while (x<=n-i-j)               // maintain heap.
			{
				child = x<<1;
				if (child>n-i-j)	break;
				if (child+1<=n-i-j && a[child+1]<a[child])	child++;
				if (a[n-i-j+1]<a[child])	break;
				a[x] = a[child];
				x = child;
			}
			a[x] = a[n-i-j+1];
		}
		
		a[n-i] = tmp;	x = n-i;           // put back into the heap.
		while (x>1 && tmp<a[x>>1])
		{
			a[x] = a[x>>1];
			x >>= 1;
		}
		a[x] = tmp;
		total += tmp;
	}
	
	total = (long long)((p*total+99.)/100.); 
	printf("%d",total);
	
	return 0;
}
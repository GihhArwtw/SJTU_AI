#include<cstdio>
using namespace std;
#define INF 1e11

int main()
{
	int n;
	scanf("%d",&n);
	int* a = new int[n+5];
	for (int i=1; i<=n; i++)	scanf("%d",&a[i]);
	
	int top=1;
	int* heap = new int[n+5];
	heap[top] = a[n];
	long long profit = 0;
	for (int i=n-1; i>=1; i--)
	{
		if (a[i]<heap[1])
		{
			profit = profit + heap[1] - a[i];
			int x = 1;
			while (x*2<=top)
			{
				int child = x*2;
				if (x*2+1<=top && heap[x*2+1]>heap[x*2])	child++;
				if (heap[child]<=a[i]) break;
				heap[x] = heap[child];
				x = child;
			}
			heap[x] = a[i];
		}
		int x=++top;
		while (x/2 && heap[x/2]<a[i])
		{
			heap[x] = heap[x/2];
			x = x/2;
		}
		heap[x] = a[i];
	}
	
	printf("%lld",profit);
	
	return 0;
}
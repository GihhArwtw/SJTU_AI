#include <cstdio>
#define INF 536870912
using namespace std;

int* a;

int search(int* g, int num, int l, int r)
{
	if (l==r) return l;
	int mid = (l+r)/2;
	if (a[g[mid]]==num)		return mid;
	if (a[g[mid]]<num)		return search(g,num,mid+1,r);
	if (a[g[mid]]>num)		return search(g,num,l,mid);
	return -1;
}

void traverse(int* prev, int q)
{
	if (prev[q]==0)
	{
		printf("%d",a[q]);
		return;
	}
	traverse(prev, prev[q]);
	printf(" %d",a[q]);
}

int main()
{
	int n;
	scanf("%d",&n);
	
	a = new int[n+5];
	int* g = new int[n+5];
	int* prev = new int[n+5];
	
	for (int i=0; i<=n; i++)	g[i] = 0;
	a[0] = INF;
	
	int max = 0, q = -1;
	for (int i=1; i<=n; i++)
	{
		scanf("%d",&a[i]);
		int j = search(g, a[i], 1, i+1);
		if (j>max)
		{
			max = j;
			q = i;
		}
		else if (j==max && a[i]<=a[q])
		{
			q = i;
		}
		prev[i] = g[j-1];
		g[j] = i;
	}
	
	printf("%d\n",max);
	
	traverse(prev, q);
	return 0;
}
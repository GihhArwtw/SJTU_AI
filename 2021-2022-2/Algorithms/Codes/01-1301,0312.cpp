#include<cstdio>
using namespace std;

void mergesort(int l, int r, int* a, int* cnt)
{
	if (l==r)  return;
	int mid = (l+r)/2;
	mergesort(l,mid,a,cnt);
	mergesort(mid+1,r,a,cnt);
	
	int *b = new int [r+1];
	int i;
	for (i=l; i<=r; i++)	b[i] = a[i];
	
	i = l;	
	int j = mid+1, k = l;
	while (i<=mid && j<=r)
	{
		if (b[i]<=b[j])
		{
			a[k] = b[i];
			cnt[a[k]] += j-mid-1; 
			i++;
		}
		else if (b[i]>b[j])
		{
			a[k] = b[j];
			j++;
			cnt[a[k]] += mid-i+1;
		}
		k++;
	}
	
	while (i<=mid)
	{
		a[k] = b[i];
		cnt[a[k]] += r-mid;
		i++;	k++;
	}
	
	while (j<=r)
	{
		a[k] = b[j];
		j++;	k++;
	}
	
	delete[] b;
}

int main()
{
	int n;
	scanf("%d",&n);
	int *a = new int[n];
	for (int i=0; i<n; i++)		scanf("%d",&a[i]);
	int *cnt = new int[n+10];
	for (int i=1; i<=n; i++)	cnt[i] = 0;
	mergesort(0,n-1,a,cnt);
	for (int i=1; i<=n; i++)	printf("%d ",cnt[i]);
	return 0;
}
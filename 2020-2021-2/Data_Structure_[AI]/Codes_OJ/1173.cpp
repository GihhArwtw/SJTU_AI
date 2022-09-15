#include <iostream>
using namespace std; 

void mergesort(int a[], int l, int r)
{
	if (l==r) return;
	int mid = (l+r)/2;
	mergesort(a,l,mid);
	mergesort(a,mid+1,r);
	
	int* tmp = new int[r-l+1];
	int i = l, j = mid+1, k = 0;
	
	while (i<=mid && j<=r)
	{
		if (a[i]>a[j])  tmp[k++] = a[i++];
		else            tmp[k++] = a[j++];
	} 
	while (i<=mid)  tmp[k++] = a[i++];
	while (j<=r)    tmp[k++] = a[j++];
	
	for (int i=l; i<=r; i++)  a[i] = tmp[i-l];
}

int main()
{
	//freopen("1173.txt","r",stdin);
	//freopen("o.txt","w",stdout);
	int n;
	scanf("%d",&n);
	
	int* a = new int[n+1], * b = new int[n+1];
	for (int i=0; i<n; i++) scanf("%d",&a[i]);
	for (int i=0; i<n; i++) scanf("%d",&b[i]);
	mergesort(a,0,n-1);
	mergesort(b,0,n-1);
	
	long long max = 0, min = 0;
	for (int i=0; i<n; i++)
	{
		max += (long long)a[i]*(long long)b[i];
		min += (long long)a[i]*(long long)b[n-1-i];
	}
	
	printf("%lld %lld",max,min);
	
	fclose(stdin);
	fclose(stdout);
	return 0;
}

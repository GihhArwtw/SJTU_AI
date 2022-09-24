#include <cstdio>
#define MAX 2147483647
using namespace std;

int main()
{
	int n;
	scanf("%d",&n);
	int* a = new int[n+5];
	for (int i=0; i<n; i++)  scanf("%d",&a[i]);
	
	int min, minn, p, q, total = 0;
	for (int j=1; j<n; j++)
	{
		min = minn = MAX;
		p = q = n+1;
		for (int i=0; i<n; i++)
			{
				if (a[i]<min) { minn = min;	q = p; min = a[i]; p = i; }
				else if (a[i]<minn) { minn = a[i]; q = i; } 
			}
		total += (minn+min);
		a[p] = minn+min;
		a[q] = MAX;
	} 
	printf("%d",total);
	
	return 0;
}

#include <cstdio>
using namespace std;

int main()
{
	int n, k;
	scanf("%d %d",&n,&k);
	/*
	if (k==500000)
	{
		printf("75667/75677\n");
		return 0;
	 } 
	*/
	
	int count = 1;
	int* f = new int[n+10];
	int* ord = new int[k+10];
	for (int i=2; i<=n; i++)  { f[i] = i-1; }
	ord[0] = n;
	
	int t = 0, maxa, maxb, q;
	for (int i=0; i<k; i++)
	{
		maxa = 0; maxb = 1; q = -1;
		for (int j=0; j<count; j++)
			if (maxa*ord[j]<maxb*f[ord[j]])
			{
				maxa = f[ord[j]];
				maxb = ord[j];
				q = j;
			}
		if (ord[q]==n)  ord[count++] = n; 
		if (count>n) count = n;
		if (i==k-1) break;
		f[ord[q]]--;
		ord[q]--;
		for (int j=0; j<count; j++)
			if (maxa*ord[j] == maxb*f[ord[j]])
			{
				if (ord[j]==n)  ord[count++] = n;
				f[ord[j]]--;
				ord[j]--;
			}
	}
	
	printf("%d/%d",f[ord[q]],ord[q]);
	 
	 
	delete []f;
	return 0;
}

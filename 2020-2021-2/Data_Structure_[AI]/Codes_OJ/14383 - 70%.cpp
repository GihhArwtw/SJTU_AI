#include <cstdio>
using namespace std;

int main()
{
	int n, k;
	scanf("%d %d",&n,&k);
	
	int** a = new int*[2], **b = new int*[2];
	a[0] = new int [k+10];
	b[0] = new int [k+10];
	a[1] = new int [k+10];
	b[1] = new int [k+10];
	int count = 0;
	
	int t = 0;
	for (int i=0; i<k+10; i++) { a[0][i] = 0; b[0][i] = 1; }
	
	for (int i=2; i<=n; i++)
	{
		int ii = 0 , j = i-1, tmp = count;
		count = 0;
		while (j && ii<tmp && count<k)
		{
			while (ii<tmp && count<k && a[t][ii]*i>=b[t][ii]*j)
			{
				a[1-t][count] = a[t][ii];
				b[1-t][count] = b[t][ii];
				ii++; count++;
			}
			if (a[t][ii]*i == b[t][ii]*j) { j--; continue;}
			if (!j) break;
			if (count>=k) break; 
			a[1-t][count] = j;
			b[1-t][count] = i; 
			count++;  j--;
		}
		while (j && count<k)
		{
			a[1-t][count] = j; b[1-t][count] = i; count++; j--;
		}
		while (ii<tmp && count<k)
		{
			a[1-t][count] = a[t][ii];
			b[1-t][count] = b[t][ii];
			count++; ii++;
		}
		t = 1-t;
	}
	
	printf("%d/%d",a[t][k-1],b[t][k-1]);
	
	
	delete []a[0];
	delete []a[1];
	delete []b[0];
	delete []b[1];
	 
	return 0;
}

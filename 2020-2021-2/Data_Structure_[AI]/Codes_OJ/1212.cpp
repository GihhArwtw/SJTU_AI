#include <iostream>
#include <cstdio>
using namespace std;

int main()
{
	int n,m;
	cin >> n >> m;
	
	int** a = new int*[n];
	for (int i=0; i<n; i++)
	{
		a[i] = new int[n];
		for (int j=0; j<n; j++)  a[i][j] = 1000000000;
		a[i][i] = 0;
	}
	
	int x,y;
	for (int i=0; i<m; i++)
	{
		cin >> x >> y;
		a[x-1][y-1] = 1; a[y-1][x-1] = 1;
	}
	
	for (int k=0; k<n; k++)
	{
		for (int i=0; i<n; i++)
		{
			for (int j=0; j<n; j++)
				if (a[i][j]>a[i][k]+a[k][j])  a[i][j] = a[i][k]+a[k][j];
		}
	}
	
	double rate;
	for (int i=0; i<n; i++)
	{
		cout << (i+1) << ": ";
		rate = 0.0;
		for (int j=0; j<n; j++)
			if (a[i][j]<=6)  rate = rate+1.0/n;
		delete []a[i];
		
		rate *= 100;
		
		printf("%.2f%\n",rate);
	}
	
	
	delete []a;
	
	return 0;
}

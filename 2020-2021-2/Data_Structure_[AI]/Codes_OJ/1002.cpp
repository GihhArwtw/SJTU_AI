#include <iostream>
using namespace std;

int main()
{
	int n,m;
	int** a;
	int i,j,x;
	
	cin >> n >> m;
	a = new int* [n+1];
	for (i=0; i<=n; i++)  { a[i] = new int[m+1]; a[i][0] = 0; }
	
	for (i=0; i<=m; i++)  a[0][i] = 0;
	for (i=1; i<=n; i++)
	{
		for (j=1; j<=m; j++)
		{
			cin >> x;
			a[i][j] = a[i-1][j] + a[i][j-1] - a[i-1][j-1] + x; 
		}
		
	}
	
	int y, max = 0, total;
	cin >> x >> y;
	for (i=0; i<=n-x; i++)
		for (j=0; j<=m-y; j++)
			{
				total = a[i+x][j+y] - a[i+x][j] - a[i][j+y] + a[i][j];
				if (total>max)  max = total;
			}
	cout << max;
	
	for (i=0; i<=n; i++)  delete []a[i];
	delete []a;
	
	return 0;
}

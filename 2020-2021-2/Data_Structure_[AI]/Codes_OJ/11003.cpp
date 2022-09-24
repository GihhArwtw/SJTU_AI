#include <iostream>

using namespace std;

int main()
{
	int n;
	cin >> n;
	
	int** a = new int* [n+2], **b = new int* [n+2];
	for (int i=0; i<=n+1; i++) { a[i] = new int[n+2]; b[i] = new int[n+2]; } 
	
	int i,j;
	for (i=1; i<=n; i++)
		for (j=1; j<=n; j++)  cin >> a[i][j];
	
	int step = 0; 
	bool flag;
	while (true)
	{
		flag = false;
		for (i=1; i<=n; i++)
		{
			for (j=1; j<=n; j++)
			{
				b[i][j] = a[i][j];
				if (a[i][j]==0)	  flag = true;
			} 
		}
		if (!flag) break;
		for (i=1; i<=n; i++)
		{
			for (j=1; j<=n; j++)
			{
				if (b[i][j]==1)
				{
					if (!b[i-1][j])  a[i-1][j] = 1;
					if (!b[i][j-1])  a[i][j-1] = 1;
					if (!b[i+1][j])  a[i+1][j] = 1;
					if (!b[i][j+1])  a[i][j+1] = 1;
				}
			}
		}
		
		step++;
	}
	
	cout << step;
	
	return 0;
}

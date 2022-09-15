#include <iostream>
using namespace std;

int main()
{
	long long n;
	const int MOD = 2010;

	/*
	bool **a = new bool* [2010];
	for (int i=0; i<2010; i++) 
	{
		a[i] = new bool [2010];
		for (int j=0; j<2010; j++)  a[i][j] = false;
	}
	int x,y,z;
	x = 1; y = 0; //x:f[-1], y:f[0]
	long long i;
	for (i = 1; i<=210000000000; i++)
		{
			z = (x+y)%MOD;
			x = y;
			y = z;
			if (a[x][y]) break;
			a[x][y] = true;
		}	
	cout << i << " " << x << " " << y;
	*/
	
	int x, y, z;
	
	cin >> n;
	n = n % 2040;
	x = 1;
	y = 0;
	for (int i=1; i<=n; i++)
	{
		z = (x+y)%MOD;
		x = y;
		y = z;
	}
	cout << y;
	
}

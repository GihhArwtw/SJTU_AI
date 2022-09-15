#include <iostream>
using namespace std;

int main()
{
	int n,m;
	cin >> n >> m;
	int* f = new int[n+3];
	for (int i=1; i<=n; i++)  f[i] = i;
	
	int x, y, z;
	for (int i=0; i<m; i++)
	{
		cin >> z >> x >> y;
		if (z==1)
		{
			int fx = x, fy = y;
			while (f[fx]!=fx)  fx = f[fx];
			while (f[fy]!=fy)  fy = f[fy];
			z = f[fy] = fx;
			
			fx = x; fy = y; int tmp = 0;
			while (f[fx]!=fx) { tmp = f[fx]; f[fx] = z; fx = tmp;	}
			while (f[fy]!=fy) { tmp = f[fy]; f[fy] = z; fy = tmp;   }
		}
		else
		{
			while (f[x]!=x)  x = f[x];
			while (f[y]!=y)  y = f[y];
			if (f[x]==f[y])  cout << "Y\n";
			else             cout << "N\n";
		}
	}
	
	return 0;
}

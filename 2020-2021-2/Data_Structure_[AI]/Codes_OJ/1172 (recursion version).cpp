#include <iostream>
using namespace std;

int merge(int x, int y, int* &f)
{
	if (f[x]==x && f[y]==y)  { f[y] = x; return x; }
	return (f[x]=f[y]=merge(f[x],f[y],f));
}

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
		if (z==1)  merge(f[x],f[y],f);
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

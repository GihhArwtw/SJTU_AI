#include<cstdio>
using namespace std;

int read_string(char* a)
{
	char c = getchar();
	while (!(c>='a' && c<='z'))	c = getchar();
	int n=0;
	while (c>='a' && c<='z')
	{
		a[n++] = c;
		c = getchar();
	}
	return n;
}

inline int min(int x, int y)
{
	if (x<y)	return x;
	return y;
}

int main()
{
	int x,y;
	scanf("%d %d",&x,&y);
	
	char* a = new char[3010], *b = new char[3010];
	int m = read_string(a), n = read_string(b);
	
	int** f = new int*[m+5];
	for (int i=0; i<=m; i++)
	{
		f[i] = new int[n+5];
		f[i][0] = i*x;
	}
	
	for (int i=0; i<=n; i++)	f[0][i] = i*x; 
	
	
	for (int i=1; i<=m; i++)
	{
		for (int j=1; j<=n; j++)
		{
			f[i][j] = f[i-1][j-1];
			if (a[i-1]!=b[j-1])		f[i][j] = f[i][j]+y;
			f[i][j] = min(f[i][j],min(f[i-1][j]+x,f[i][j-1]+x));
		}
	}
	
	printf("%d",f[m][n]);
	
	return 0;
}
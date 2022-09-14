#include <cstdio>
using namespace std;
#define MOD 1000000007

int main()
{
	int m,n,k;
	scanf("%d %d %d",&m,&n,&k);
	
	bool** a = new bool*[m+5];
	for (int i=1; i<=m; i++)	a[i] = new bool[n+5];
	for (int i=1; i<=m; i++)
	{
		for (int j=1; j<=n; j++)	a[i][j] = true;
	}
	
	int x,y;
	for (int i=0; i<k; i++)
	{
		scanf("%d %d",&x,&y);
		a[x][y] = false;
	}
	
	int states = 1<<n;
	int*** f = new int**[2];
	for (int j=0; j<2; j++)
	{
		f[j] = new int*[states];	
		for (int i=0; i<states; i++)	f[j][i] = new int[states];
	}
	
	int flag = 1;
	int curr=0, next=0;
	for (int k=1; k<=n; k++)
	{
		curr = curr | ( (1-a[1][k]) << (n-k) );
		next = next | ( (1-a[2][k]) << (n-k) );
	}
		
	// We place the first two columns.
	for (int i=0; i<states; i++)
	{
		for (int j=0; j<states; j++)
		{
			if( (i&curr) || (j&next) || ((i<<2)&j) || ((j<<2)&i) )
			{	f[flag][i][j] = 0;	}
            else
			{	f[flag][i][j] = 1; 	}
		}
	}
	
	for (int l=2; l<=m-1; l++)
	{
		flag = 1-flag;	
		curr = 0;
		next = 0;
		for (int k=1; k<=n; k++)
		{
			curr = curr | (  (1-a[l][k])  << (n-k) );
			next = next | ( (1-a[l+1][k]) << (n-k) );
		}
		for (int i=0; i<states; i++)
		{
			for (int j=0; j<states; j++)
			{
				if( (i&curr) || (j&next) || ((i<<2)&j) || ((j<<2)&i) )
				// conflict!
				{	f[flag][i][j] = 0;	}
	            else
				{
					f[flag][i][j] = 0;
					for (int k=0; k<states; k++)
					{
						if ( ( (j<<1) & k) || ( (k<<1) & j ) )	continue;
						// conflict!
						f[flag][i][j] = (f[flag][i][j] + f[1-flag][k][i]) % MOD;
					}
				}
			}
		}
	}
	
	int ans = 0;
	for (int i=0; i<states; i++)
	{
		for (int j=0; j<states; j++)
		{
			ans = (ans+f[flag][i][j]) % MOD;
		}
	}
	
	printf("%d",ans);
	
	return 0;
}
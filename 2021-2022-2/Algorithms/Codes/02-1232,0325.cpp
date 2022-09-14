#include <cstdio>
using namespace std;
#define INF 536870911    //2147483647/4

int main()
{
	int n,m;
	scanf("%d %d",&n,&m);
	
	int* a = new int[m+10];
	int* b = new int[m+10];
	int* w = new int[m+10];
	
	for (int i=0; i<m; i++)		scanf("%d %d %d",&a[i],&b[i],&w[i]);
	
	int* dist = new int[n+10];
	for (int i=1; i<=n; i++)	dist[i] = INF;
	for (int i=0; i<n-1; i++)
	{
		for (int i=0; i<m; i++)
		{
			if (dist[b[i]]>dist[a[i]]+w[i])
				dist[b[i]]=dist[a[i]]+w[i];
		}
	}
	
	for (int i=0; i<m; i++)
		if (dist[b[i]]>dist[a[i]]+w[i])
		{
			printf("Yes");
			return 0;
		}
	
	printf("No");
	return 0;
}
#include <cstdio>
using namespace std;
#define INF 461168618427387903

long long** a;
int n,m,s,t;
int*  pre;

inline long long min(long long x, long long y)
{
	if (x<y)	return x;
	return y;
}

bool BFS()
{
	bool reachable = false;
	int* queue = new int[n+10], * dist = new int[n+10];
	pre = new int[n+10];
	
	int head = 0, tail = 0;
	queue[head] = s;	pre[s] = -1;
	
	for (int i=1; i<=n; i++)	dist[i] = -1;
	dist[s] = 0;
	
	while (head<=tail)
	{
		int u=queue[head];
		for (int v=1; v<=n; v++)
			if (a[u][v]>0 && dist[v]<0)
			{
				tail++;
				queue[tail] = v;
				dist[v] = dist[u]+1;
				pre[v] = u;
				if (v==t)
				{
					reachable = true;
					break;
				}
			}
		head++;
	}
	delete []queue;
	delete []dist;
	
	return reachable;
}

long long augment ()
{
	int u = t;
	long long minflow = INF;
	while (u!=s)
	{
		minflow = min(minflow,a[pre[u]][u]);
		u = pre[u];
	}
	
	u = t;
	while (u!=s)
	{
		a[pre[u]][u] -= minflow;
		a[u][pre[u]] += minflow;
		u = pre[u];
	}
	
	return minflow;
}

int main()
{
	scanf("%d %d %d %d",&n,&m,&s,&t);
	
	a = new long long*[n+10];
	for (int i=1; i<=n; i++)
	{
		a[i] = new long long[n+10];
		for (int j=1; j<=n; j++)  a[i][j] = 0;
	}	
	
	int x,y,w;
	for (int i=0; i<m; i++)
	{
		scanf("%d %d %d",&x,&y,&w);
		a[x][y] += w;
	}
	
	
	long long flow = 0;
	while (BFS())
	{
		flow = flow + augment();
	}
	
	printf("%lld",flow);
	
	delete[] pre;
	for (int i=1; i<=n; i++)	delete[] a[i];
	delete[] a;
	
	return 0;
}
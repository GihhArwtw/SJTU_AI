#include <cstdio>
using namespace std;
#define INF 1073741824

int** a;
int n,m,s,t;
int*  pre;

inline int min(int x, int y)
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
	
	for (int i=0; i<=n+1; i++)	dist[i] = -1;
	dist[s] = 0;
	
	while (head<=tail)
	{
		int u=queue[head];
		for (int v=0; v<=n+1; v++)
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

int augment ()
{
	int u = t;
	int minflow = INF;
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
	scanf("%d %d",&m,&n);
	s = 0;
	t = n+1;
	
	int* pig = new int[m+10];
	for (int i=1; i<=m; i++)	scanf("%d",&pig[i]);
	
	a = new int*[n+10];
	for (int i=0; i<=n+1; i++)
	{
		a[i] = new int[n+10];
		for (int j=0; j<=n+1; j++)  a[i][j] = 0;
	}	
	
	int keys, id;
	int* last = new int[m+10];
	for (int i=1; i<=m; i++)	last[i] = -1;
	for (int i=1; i<=n; i++)
	{
		scanf("%d",&keys);
		for (int j=0; j<keys; j++)
		{
			scanf("%d",&id);
			if (last[id]<0)		a[s][i] += pig[id];
			else				a[last[id]][i] = INF;
			last[id] = i;
		}
		scanf("%d",&id);
		a[i][t] = id;
	}
	
	int flow = 0;
	
	while (BFS())
	{
		flow = flow + augment();
	}
	
	printf("%d",flow);
	
	delete[] pre;
	for (int i=1; i<=n; i++)	delete[] a[i];
	delete[] a;
	
	return 0;
}
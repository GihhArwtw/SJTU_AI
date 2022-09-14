#include <cstdio>
using namespace std;
#define INF 1000

int** a;
int N,s,t;
int*  pre;

inline int min(int x, int y)
{
	if (x<y)	return x;
	return y;
}

bool BFS()
{
	bool reachable = false;
	int* queue = new int[N+10], * dist = new int[N+10];
	pre = new int[N+10];
	
	int head = 0, tail = 0;
	queue[head] = s;	pre[s] = -1;
	
	for (int i=0; i<=N+1; i++)	dist[i] = -1;
	dist[s] = 0;
	
	while (head<=tail)
	{
		int u=queue[head];
		for (int v=0; v<=N+1; v++)
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
	int L;
	scanf("%d",&L);
	
	int n,m,k,id;
	for (int l=0; l<L; l++)
	{
		scanf("%d %d %d",&n,&m,&k);
		
		N = n*m;	s = 0;		t = N+1;
		
		int x,y;
		if (N-k%2==1)
		{
			for (int i=0; i<k; i++)	scanf("%d %d",&x,&y);
			printf("NO\n");
			continue;
		}
		
		a = new int*[N+10];
		for (int i=0; i<=N+1; i++)
		{
			a[i] = new int[N+10];
			for (int j=0; j<=N+1; j++)	a[i][j] = 0;
		}
		
		for (int i=0; i<k; i++)
		{
			scanf("%d %d",&x,&y);
			a[s][(x-1)*m+y] = -1;        // just a label.
		}
		
		for (int i=1; i<=n; i++)
		{
			for (int j=1; j<=m; j++)
			{
				id = (i-1)*m+j;
				if (a[s][id]<0)
				{
					a[s][id] = 0;
					a[id][t] = 0;
					continue;
				}
				if ((i+j)%2==1)
				{
					a[id][t] = 1;
					continue;
				}
				a[s][id] = 1;
				if (j<m)	a[id][id+1] = INF;
				if (j>1)	a[id][id-1] = INF;
				if (i<n)	a[id][id+m] = INF;
				if (i>1)	a[id][id-m] = INF;
			}
		}
		
		int flow = 0;
		while (BFS())
		{
			flow = flow + augment();
		}
	
		if (flow*2 == N-k)
		{
			printf("YES\n");
		}
		else
		{
			printf("NO\n");
		}
	
		delete[] pre;
		for (int i=1; i<=n; i++)	delete[] a[i];
		delete[] a;
	}
	
	return 0;
}
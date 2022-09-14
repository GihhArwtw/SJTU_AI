#include <cstdio>
using namespace std;
#define INF 536870911   // 2147483647/4

struct link{
	int node;
	link* next;
	link(const int x, link* y=NULL): node(x), next(y) {}
	link(): next(NULL) {}
};

int main()
{	
	int n,m,s,t;
	scanf("%d %d %d %d",&n,&m,&s,&t);
	
	int u,v,w;
	link** a;           // a: neighbors to which weight = 0.
	a = new link*[n+10];
	for (int i=1; i<=n; i++)	a[i] = new link;
	
	int count = 0;
	int* st = new int[m+10];
	int* en = new int[m+10];
	for (int i=0; i<m; i++)
	{
		scanf("%d %d %d",&u,&v,&w);
		if (w==0)
		{
			link* p = new link(v, a[u]->next);
			a[u]->next = p;
			link* q = new link(u, a[v]->next);
			a[v]->next = q;
		}
		else
		{
			st[count] = u;
			en[count] = v;
			count++;
		}
	}
	m = count;
	
	// Find all connected components in 0-weighted subgraph.
	
	int* con = new int[n+10];
	for (int i=1; i<=n; i++)	con[i] = -1;
	
	count = 0;
	int* queue = new int[n+10];
	int head=0, tail=0;
	for (int i=1; i<=n; i++)
		if (con[i]<0)
		{
			count++;
			con[i] = count;
	
			head = tail = 0;
			queue[head] = i;
			while (head<=tail)
			{
				u = queue[head];
				link* p = a[u]->next;
				while (p)
				{
					v = p->node;
					if (con[v]<0)        // not visited
					{
						tail++;
						queue[tail] = v;
						con[v] = con[u];
					}
					p = p->next;
				}
				head++;
			}
		}
	
	if (con[s]==con[t])
	{
		printf("0");
		return 0;
	}
	
	for (int i=0; i<n+10; i++)	delete[] a[i];
	delete[] a;
	delete[] queue;
	
	// Extract the metagraph.
	a = new link*[count+10];
	for (int i=1; i<=count; i++)		a[i] = new link;
	for (int i=0; i<m; i++)
	{
		u = con[st[i]];
		v = con[en[i]];
		link* p = new link(v, a[u]->next);
		a[u]->next = p;
		link* q = new link(u, a[v]->next);
		a[v]->next = q;
	}
	
	int* dist = new int[count+10];
	for (int i=1; i<=count; i++)  dist[i] = INF;
	
	queue = new int[count+10];
	head = 0;	tail = 0;
	queue[head] = con[s];
	dist[con[s]] = 0;
	while (head<=tail)
	{
		u = queue[head];
		link* p = a[u]->next;
		while (p)
		{
			v = p->node;
			if (dist[v]>n)        // not visited
			{
				tail++;
				queue[tail] = v;
				dist[v] = dist[u]+1;
				if (v == con[t])
				{
					printf("%d",dist[v]);
					return 0;
				}
			}
			p = p->next;
		}
		head++;
	}
	// I suppose this section of the program would never be visited.
	return 0;
}
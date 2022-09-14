#include<cstdio>
using namespace std;
#define INF 536870911       // 2147483647/4

struct link{
	int node;
	int weight;
	link* next;
	link(int x=0, int w=0, link* y=NULL): node(x), weight(w), next(y) {}
};

int main()
{
	int n,m,s,t;
	scanf("%d %d %d %d",&n,&m,&s,&t);
	
	link** a = new link*[n+10];
	for (int i=1; i<=n; i++)	a[i] = new link;
	
	int x,y,z;
	for (int i=0; i<m; i++)
	{
		scanf("%d %d %d",&x,&y,&z);
		link* p = new link(y,z,a[x]->next);
		a[x]->next = p;
		link* q = new link(x,z,a[y]->next);
		a[y]->next = q;
	}
	
	int* dist = new int[n+10];     // binary heap
	int* order = new int[n+10];    // dist[i] is the distance from s to order[i]
	int* place = new int[n+10];    // the distance from s to i is dist[place[i]]
	
	dist[1] = 0;	order[1] = s;	place[s] = 1;
	for (int i=2; i<=s; i++)
	{
		dist[i] = INF;	order[i] = i-1;	place[i-1] = i;
	}
	for (int i=s+1; i<=n; i++)
	{
		dist[i] = INF;	order[i] = i;	place[i] = i;
	}
	
	int currdist, tmpdist, tmpord;
	for (int i=1; i<n; i++)
	{
		x = order[1];	currdist = dist[1];			// pop the min
		if (x==t)	break;
		
		tmpord = order[n-i+1];	order[n-i+1] = order[1]; order[1] = tmpord;
		place[x] = n-i+1;	place[order[1]] = 1;
		tmpdist = dist[n-i+1]; dist[n-i+1] = dist[1]; dist[1] = tmpdist;
		y = 1;
		while (y<=n-i)
		{
			z = y<<1;
			if (z>n-i)	break;
			if (z+1<=n-i && dist[z+1]<dist[z])	z++;
			if (tmpdist<=dist[z])	break;
			dist[y] = dist[z];
			order[y] = order[z];
			place[order[y]] = y;
			y = z;
		}
		dist[y] = tmpdist;
		order[y] = tmpord;
		place[order[y]] = y;
		// maintain the heap.
		
		link* p = a[x]->next;
		while (p)
		{
			y = p->node;
			if (dist[place[y]] > currdist + p->weight)
			{
				tmpdist = currdist + p->weight;       // update.
				z = place[y];
				while (z>1 && tmpdist<dist[z>>1])     // maintain the heap.
				{
					dist[z] = dist[z>>1];
					order[z] = order[z>>1];
					place[order[z]] = z;
					z >>= 1;
				}
				dist[z] = tmpdist;
				order[z] = y;
				place[y] = z;
			}
			p = p->next;
		}
	}
	
	printf("%d",dist[place[t]]);
	
	return 0;
}
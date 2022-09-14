#include <cstdio>
using namespace std;

struct link{
	int node;
	link* next;
	link(int x=0, link* y=NULL): node(x), next(y) {}
};

int main()
{
	int n,m,r;
	scanf("%d %d %d",&n,&m,&r);
	
	link** a = new link*[n+10];
	for (int i=1; i<=n; i++)	a[i] = new link;
	
	int x,y;
	int* indeg = new int[n+10];
	for (int i=1; i<=n; i++)	indeg[i] = 0;
	for (int i=0; i<m; i++)
	{
		scanf("%d %d",&x,&y);
		link* p = new link(y,a[x]->next);
		a[x]->next = p;
		indeg[y]++;
	}
	
	int* queue = new int [n+10];
	int head=0, tail=-1;
	for (int i=1; i<=n; i++)
		if (indeg[i]==0)
		{
			tail++;
			queue[tail] = i;
		}
	
	while (head<=tail)
	{
		x = queue[head];
		link* p = a[x]->next;
		while (p)
		{
			indeg[p->node]--;
			if (!indeg[p->node])
			{
				tail++;
				queue[tail] = p->node;
			}
			p = p->next;
		}
		head++;
	}
	
	int* order = new int[n+10];
	for (int i=0; i<n; i++)		order[queue[i]] = i;
	
	for (int i=0; i<r; i++)
	{
		scanf("%d %d",&x,&y);
		if (order[x]<=order[y])
		{
			printf("%d %d\n",x,y);
		}
		else
		{
			printf("%d %d\n",y,x);
		}
	}
	
	return 0;
}
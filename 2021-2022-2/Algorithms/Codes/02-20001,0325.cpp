#include<cstdio>
using namespace std;

struct link
{
	int node;
	link* next;	
	link(int x=0, link* y=NULL): node(x), next(y) {}
};

int count = 0;

void DFS(link** graph, int s, int* post, bool* visit)
{
	link* p = graph[s]->next;
	visit[s] = true;
	while (p)
	{
		if (!visit[p->node])
			DFS(graph, p->node, post, visit);
		p = p->next;
	}
	post[++count] = s;
}

int SSC(link** graph, int s, bool* visit, int* conn)
{
	link* p = graph[s]->next;
	visit[s] = true;
	while (p)
	{
		if (!visit[p->node])
		{
			conn[p->node] = conn[s];
			SSC(graph, p->node, visit, conn);
		}
		else
		{
			if (conn[p->node] != conn[s])      // the out-degree of the current component is not 0.
			{
				count = -1;
				return -1;
			}
		}
		if (count==-1)	return -1;
		p = p->next;
	}
	count++;
	return count;
}

int main()
{
	int n,m;
	scanf("%d %d",&n,&m);
	
	link** a = new link*[n+10];
	link** b = new link*[n+10];
	for (int i=1; i<=n; i++)
	{
		a[i] = new link;
		b[i] = new link;
	}
	
	int x,y;
	for (int i=0; i<m; i++)
	{
		scanf("%d %d",&x,&y);
		link* p = new link(y,a[x]->next);
		a[x]->next = p;
		link* q = new link(x,b[y]->next);
		b[y]->next = q;
	}
	
	int* post = new int[n+10];
	bool* visit = new bool[n+10];
	int* conn = new int[n+10];
	for (int i=1; i<=n; i++)
	{
		post[i] = 0;
		visit[i] = false;
		conn[i] = i;
	}
	
	count = 0;
	for (int i=1; i<=n; i++)
		if (!visit[i])	DFS(b,i,post,visit);
			
	count = 0;
	
	// We find the strongly connected component whose out-degree is 0.
	int num=0;
	for (int i=1; i<=n; i++)	visit[i] = false;
	for (int i=n; i>=1; i--)
		if (conn[post[i]]==post[i])
		{	
			SSC(a,post[i],visit,conn);
			if (count==-1)	count = num;
			num = count;
		}
	
	printf("%d",n-num);
	return 0;
}
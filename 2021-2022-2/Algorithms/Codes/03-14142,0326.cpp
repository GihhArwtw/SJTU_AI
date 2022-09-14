#include<cstdio>
using namespace std;

struct link{
	int node;
	int place;
	link* next;
	link(int x=0, int w=0, link* y=NULL): node(x), place(w), next(y) {}
};

inline int min(int x, int y)
{
	if (x<y) return x;
	return y;
}

link**  a;
int*  ord;
int* conn;
int* x, * y;
long long * w, * backup;

void tarjan(int s, int edge)
{
	static int count=0;
	ord[s] = conn[s] = ++count;
	link* p = a[s]->next;
	while (p)
	{
		if (!ord[p->node])
		{
			tarjan(p->node,p->place);
			conn[s] = min(conn[s], conn[p->node]);
			if (conn[p->node]>ord[s])	     	// is a cut-edge
			{
				w[p->place] = backup[p->place];
			}
		}
		else
		{
			if (p->place != edge)
			{
				conn[s] = min(conn[s], ord[p->node]);
			}
		}
		p = p->next;
	}
}

void sort(int l, int r)
{
	if (l>=r)	return;
	long long pivot = w[l];
	int tmpx = x[l], tmpy = y[l];
	int ii=l, jj=r;
	while (ii<jj)
	{
		while (jj>ii && w[jj]>pivot)	jj--;
		if (ii==jj)	break;
		w[ii] = w[jj];	x[ii] = x[jj];	y[ii] = y[jj];
		ii++;
		while (ii<jj && w[ii]<pivot)	ii++;
		if (ii==jj)	break;
		w[jj] = w[ii];	x[jj] = x[ii];	y[jj] = y[ii];
		jj--;
	}
	w[ii] = pivot;	x[ii] = tmpx;	y[ii] = tmpy;
	if (l<ii-1)		sort(l,ii-1);
	if (ii+1<r)		sort(ii+1,r);
}

int root(int x)
{
	if (x!=conn[x])
	{
		conn[x] = root(conn[x]);
		return conn[x];
	}
	return conn[x];
}

void merge(int x, int y)
{
	if (x<y)
	{
		conn[y] = conn[x];
		return;
	}
	conn[x] = conn[y];
}

int main()
{
	int n,m;
	scanf("%d %d",&n,&m);
	
	a = new link*[n+10];
	for (int i=1; i<=n; i++)	a[i] = new link;
	x = new int[m+10];
	y = new int[m+10];
	w = new long long[m+10];
	backup = new long long[m+10];
	
	for (int i=0; i<m; i++)
	{
		scanf("%d %d %lld %lld",&x[i],&y[i],&w[i],&backup[i]);
		link* p = new link(y[i],i,a[x[i]]->next);
		a[x[i]]->next = p;
		link* q = new link(x[i],i,a[y[i]]->next);
		a[y[i]]->next = q;
	}
	
	ord = new int[n+10];
	conn = new int[n+10];
	for (int i=1; i<=n; i++)
	{
		ord[i] = 0;		conn[i] = -1;
	}
	
	for (int i=1; i<=n; i++)
		if (!ord[i])	tarjan(i,-1);
		
	sort(0,m-1);
	for (int i=1; i<=n; i++)	conn[i] = i;
	int num = 0;
	long long total=0;
	for (int i=0; i<m; i++)
	{
		if (root(conn[x[i]])==root(conn[y[i]]))		continue;
		total += w[i];
		num++;
		if (num==n-1)	break;
		merge(root(x[i]),root(y[i]));
	}
	
	printf("%lld",total);
	
	return 0;
}
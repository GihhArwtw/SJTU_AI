#include <cstdio>
using namespace std;

struct node
{
	int left;
	int right;
	node(int x=0):left(0), right(0) {}
};

int main()
{
	int n, x, y, lf, rh, root;
	scanf("%d %d %d",&n,&x,&y);
	int* a = new int[n+1];
	node* tree = new node[n+1];
	int* f = new int[n+1]; //height of node
	
	for (int i=0; i<=n; i++)  a[i] = 0;
	for (int i=1; i<=n; i++)
	{
		scanf("%d %d",&lf,&rh);
		a[lf] = i;	a[rh] = i;
		if (lf) tree[i].left  = lf;
		if (rh) tree[i].right = rh;
	}
	int i;
	for (i=1; i<=n; i++)
		if (!a[i]) break;
	if (!a[i])  root = i;
		
	int* que = new int[n+2];
	int head =-1, tail = 0;
	int p, h;
	que[0] = i; f[i] = 1;
	while (head<tail)
	{
		p = que[++head];
		h = f[p];
		if (tree[p].left)
		{
			que[++tail] = tree[p].left;
			f[que[tail]] = h+1;
		}
		if (tree[p].right)
		{
			que[++tail] = tree[p].right;
			f[que[tail]] = h+1;
		}
	}
	delete []que;
	delete []tree; 
	// height calculation
	
	while (f[x]<f[y])	y = a[y];
	while (f[x]>f[y])	x = a[x];
	// reach the same level
	
	while (x!=y) { x = a[x]; y = a[y]; }
	printf("%d",x);
	
	delete []a;
	delete []f;
	return 0;
}

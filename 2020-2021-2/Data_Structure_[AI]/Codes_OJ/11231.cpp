#include <cstdio>
using namespace std;

struct node
{
	int ord;
	node* left;
	node* right;
	node(int x=0):ord(x), left(NULL), right(NULL) {}
};

int main()
{
	int n, x, y, lf, rh, root;
	scanf("%d %d %d",&n,&x,&y);
	int* a = new int[n+1];
	node** tree = new node*[n+1];
	int* f = new int[n+1]; //height of node
	
	for (int i=0; i<=n; i++)  { tree[i] = new node(i); a[i] = 0; }
	for (int i=1; i<=n; i++)
	{
		scanf("%d %d",&lf,&rh);
		a[lf] = i;	a[rh] = i;
		if (lf) tree[i]->left  = tree[lf];
		if (rh) tree[i]->right = tree[rh];
	}
	int i;
	for (i=1; i<=n; i++)
		if (!a[i]) break;
	if (!a[i])  root = i;
		
	node** que = new node*[n+2];
	int head =-1, tail = 0;
	node* p;
	int h;
	que[0] = tree[i]; f[i] = 1;
	while (head<tail)
	{
		p = que[++head];
		h = f[p->ord];
		if (p->left)
		{
			que[++tail] = p->left;
			f[p->left->ord] = h+1;
		}
		if (p->right)
		{
			que[++tail] = p->right;
			f[p->right->ord] = h+1;
		}
	}
	delete []que;
	delete []tree; 
	delete p;
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

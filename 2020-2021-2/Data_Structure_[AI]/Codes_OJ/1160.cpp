#include <cstdio>

struct node
{
	int datum;
	int height;
	node* left;
	node* right;
	node* par;
	
	node(int x, int h, node* p = NULL, node* y = NULL, node* z = NULL):
	    datum(x),height(h),par(p),left(y),right(z) {}
};

int main()
{
	int n, x, y;
	scanf("%d",&n);
	scanf("%d",&x);
	node* root = new node(x,0), *p = NULL, *q = NULL;
	for (int i=1; i<n; i++)
	{
		scanf("%d",&x);
		p = root;
		while (p)
		{
			q = p;
			if (x<p->datum)  p = p->left;
			else             p = p->right;
		} 
		if (x<q->datum)  q->left  = new node(x,q->height+1,q);
		else             q->right = new node(x,q->height+1,q);
	}
	
	scanf("%d %d",&x,&y);
	p = root;
	while (p)
	{
		if (x==p->datum)  break;
		if (x<p->datum)   p = p->left;
		if (x>p->datum)   p = p->right;
	}
	
	q = root;
	while (q)
	{
		if (y==q->datum)  break;
		if (y<q->datum)   q = q->left;
		if (y>q->datum)   q = q->right;
	}
	
	while (p->height>q->height)	 p = p->par;
	while (p->height<q->height)  q = q->par;
	while (p!=q)
	{
		p = p->par; q = q->par;
	} 
	
	printf("%d",p->datum);
	
	return 0;
} 

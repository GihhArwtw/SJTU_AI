#include <cstdio>
#include <iostream>
using namespace std;

struct node
{
	int datum;
	node* left;
	node* right;
	
	node(int x, node* y=NULL, node* z=NULL): datum(x),left(y),right(z) {}
};

void jumpToDigit(char& ch, int& x)
{
	while (ch!='-' && ch<'0' || ch>'9') ch = getchar();
	x = 0;
	int sym = 1;
	if (ch=='-')
	{
		ch = getchar(); sym = -1; 
	}
	while (ch>='0' && ch <='9')
	{
		x = x*10 + int(ch)-48;
		ch = getchar();
	}
	x = x*sym;
}

void remove(int x, node* &p)
{
	if (!p) return;
	
	if (p->datum>x)       remove(x,p->left);
	else if (p->datum<x)  remove(x,p->right);
	else
	{
		node* tmp; 
		if (!(p->left && p->right))
		{
			tmp = p;
			p = (p->left)?(p->left):(p->right); 
			delete tmp;
		}
		else
		{
			tmp = p->right;
			while (tmp->left)  tmp = tmp->left;
			p->datum = tmp->datum;
			remove(p->datum,p->right);
		}
	}
}

int find(int x, node* t, int &count)
{
	if (!t) return 1;
	
	if (t->left)   find(x,t->left,count);
	if (count==x)  return 0;
	count++;
	if (count==x)  { printf("%d\n",t->datum);  return 0; }
	if (t->right)  find(x,t->right,count);
	if (count==x)  return 0; else return 1;
}

void rmv(node* &t, int x, int y)
{
	if (!t) return;
	if (t->datum>x && t->left)        rmv(t->left,x,y);
	if (t->datum<y && t->right)       rmv(t->right,x,y);
	if (t->datum>x && t->datum<y)	  remove(t->datum,t);
}

int main()
{	
	int n;
	scanf("%d",&n);
	char ch = '\n';
	int x, y;
	
	int i=0;
	while (ch!='i')
	{
		while (ch!='\n'&&ch!='\r')  ch = getchar();
		while (ch=='\n'||ch==' '||ch=='\r')  ch = getchar();
		i++;
	}
	jumpToDigit(ch,x);
	node* root = new node(x), *p = NULL, *q = NULL;
	
	for (; i<n; i++)
	{
		while (ch=='\n'||ch==' '||ch=='\r')  ch = getchar();
		switch (ch)
		{
			case 'i': jumpToDigit(ch,x);  
					  p = root;
					  if (!p)
					  {
					  	  root = new node(x);
					  	  break;
					  }
					  while (p)
					  {
					 	  q = p;
						  if (p->datum>x)  p = p->left;
 						  else             p = p->right;
					  } 
					  if (q->datum==x) break;
					  if (q->datum>x)  q->left  = new node(x);
					  else             q->right = new node(x);
					  break;
					  
			case 'd': for (int j=0; j<6; j++)   ch = getchar();
					  if (ch==' ')
					  {
					  	  jumpToDigit(ch,x);
					  	  remove(x,root);
					  	  break;
					  }
					  jumpToDigit(ch,x);
					  jumpToDigit(ch,y);
					  
					  rmv(root,x,y);
					  break;
					  
					  /*
					  p = root;
					  while (p && (p->datum<x || p->datum>y))
					  {
					  	  if (p->datum<x)  p = p->right;
					  	  else             p = p->left;
					  }
					  if (!p) break;
					  if (p->left)  rmvleft(p->left,x);
					  if (p->right) rmvright(p->right,y);
					  
					  q = p->left;
			          while (q->right)  q = q->right;
					  p->datum = q->datum;
					  remove(p->datum,p->left);
					  break;
					  */
					  
			case 'f': for (int j=0; j<4; j++)   ch = getchar();	  
					  if (ch==' ')
					  {
						  jumpToDigit(ch,x);
						  p = root;
						  while (p)
						  {
						  	  if (p->datum==x)  break;
						  	  if (p->datum>x)   p = p->left;
						  	  else              p = p->right;
						  }
						  if (p)  printf("Y\n"); else printf("N\n");
						  break;
					  }
					  jumpToDigit(ch,x);
					  int count=0;	
					  if (!x)
					  {
					      printf("N\n"); break;
					  }
					  if (find(x,root,count)) printf("N\n");
					   
					  break;
		} 
	}
	
	return 0;
	
}

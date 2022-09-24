#include<iostream>
using namespace std;

#define scale 1000

struct node
{
	int datum;
	int height;
	node* left;
	node* right;
	
	node(int x, int h=0): datum(x), height(h), left(NULL), right(NULL) {}
}; 

struct NODE
{
	node* ptr;
	NODE* left;
	NODE* right;
	
	NODE(node* x): ptr(x), left(NULL), right(NULL) {}
};

node* insert(int x, node* &t, int h)
{
	if (!t)
	{
		t = new node(x,h);
		return t;
	}
	if (x<t->datum)  return insert(x,t->left,h+1);
	return insert(x,t->right,h+1);
}

void insert(node* x, NODE* &t)
{
	if (!t)
	{
		t = new NODE(x);
		return;
	}
	if (x->datum<t->ptr->datum)  insert(x,t->left);
	else                         insert(x,t->right);
}

void find(int x, NODE* &t, node* &res)
{
	if (x==t->ptr->datum)  { res = t->ptr; return; }
	if (x<t->ptr->datum)  
	{
		if (t->left)   find(x,t->left,res);
		else           { res = t->ptr; return; }
	}
	else
	{
		if (t->right)  find(x,t->right,res);
		else           { res = t->ptr; return; }
	}
}

void traverse(long long &total, node* &t)
{
	if (!t) return;
	total += t->height;
	traverse(total,t->left);
	traverse(total,t->right);
}

int main()
{
	int n,x;
	cin >> n;
	cin >> x;
	
	node* root = new node(x), *p = NULL;
	NODE* guide = new NODE(root); 
	for (int i=1; i<n; i++)  
	{
		cin >> x;
		find(x,guide,root);
		p = insert(x,root,root->height);
		if (p->height % scale == 0) insert(p,guide);
	}
	
	long long total = 0;
	traverse(total,guide->ptr);
	cout << total;
	
	return 0;
}

#ifndef TREE_H
#define TREE_H

#include "Queue.h"
#include "Stack.h"
#include <iostream>

template <typename eT>
inline eT max(const eT& x, const eT& y) { if (x>y) return x; return y; }

template <typename eT> 
class tree
{
	public:
		virtual ~tree() {}
		
		virtual bool Empty() const = 0;
		virtual int Size() const = 0;
		virtual int Height() const = 0;
		virtual void clear() = 0;
		virtual void LevelOrder() const = 0;
		virtual void PreOrder() const = 0;
		virtual void MidOrder() const = 0;
		virtual void PostOrder() const = 0;
};

template <typename eT>
class BTree;

#define Tree BTree

template <typename eT>
class node
{
	eT datum;
	node* left;
	node* right;
	
	friend BTree<eT>;
	node(): left(NULL), right(NULL) {}
	node(eT x, node* y=NULL, node* z=NULL): datum(x), left(y), right(z) {}
};

template <typename eT>
class BTree: public tree<eT>
{
	node<eT>* root;
	
	void clear(node<eT>* t);
	int Size(node<eT>* t) const;
	int Height(node<eT>* t) const;
	void PreOrder(node<eT>* t) const;
	void MidOrder(node<eT>* t) const;
	void PostOrder(node<eT>* t) const;
	
	public:
		BTree();
		BTree(const BTree&);
		BTree& operator=(const BTree&);
		~BTree();
		
		void Create(const eT&);
		
		bool Empty() const;
		node<eT>* Root() const;
		int Size() const;
		int Height() const;
		void clear();
		void DelSubTree(node<eT>* t);
	//	void LevelOrder() const;
		void PreOrder() const;
		void MidOrder() const;
		void PostOrder() const;
	 
};


/* ============================================ */
/* ============================================ */
/* ============================================ */
/* =========Tree.cpp (Id return 1 exit)======== */
/* ============================================ */
/* ============================================ */
/* ============================================ */

/* =============== BTree ================ */
template <typename eT>
BTree<eT>::BTree()
{
	root = NULL;
}

template <typename eT>
void BTree<eT>::clear(node<eT>* t)
{
	if (!t) return;
	clear(t->left);
	clear(t->right);
	delete t;
}

template <typename eT>
BTree<eT>::BTree(const BTree& x)
{
	root = new node<eT>(x.root->datum);
	Queue<node<eT>*> que,qux;
	node<eT>* p, q, tmp;
	
	qux.enQueue(x.root);
	que.enQueue(root);
	while (!qux.empty())
	{
		p = qux.deQueue();
		q = que.deQueue();
		if (p->left)
		{
			tmp = new node<eT>(p->left->datum);
			q->left = tmp;
			qux.enQueue(p->left);
			que.enQueue(q->left);
		}
		if (q->left)
		{
			tmp = new node<eT>(p->right->datum);
			q->right = tmp;
			qux.enQueue(p->right);
			que.enQueue(q->right);
		}
	}
}

template <typename eT>
BTree<eT>& BTree<eT>::operator=(const BTree<eT>& x)
{
	clear(root);
	
	root = new node<eT>(x.root->datum);
	LinkQueue<node<eT>*> que,qux;
	node<eT>* p, q, tmp;
	
	qux.enQueue(x.root);
	que.enQueue(root);
	while (!qux.empty())
	{
		p = qux.deQueue();
		q = que.deQueue();
		if (p->left)
		{
			tmp = new node<eT>(p->left->datum);
			q->left = tmp;
			qux.enQueue(p->left);
			que.enQueue(q->left);
		}
		if (q->left)
		{
			tmp = new node<eT>(p->right->datum);
			q->right = tmp;
			qux.enQueue(p->right);
			que.enQueue(q->right);
		}
	}
} 

template <typename eT>
BTree<eT>::~BTree()
{
	clear(root);
}

template <typename eT>
void BTree<eT>::Create(const eT& stopflag)
{
	eT e;
	Queue<node<eT>*> que;
	node<eT>* p, * tmp;
	
	(std::cin) >> e;
	if (e==stopflag) return;
	root = new node<eT>(e);
	
	que.enQueue(root);
	while (!que.empty())
	{
		p = que.deQueue();
		(std::cout) << "Sons of node " << p->datum << ": (Left Right))\n";
		(std::cin) >> e;
		if (e!=stopflag)
		{
			tmp = new node<eT>(e);
			p->left = tmp;
			que.enQueue(tmp);
		}
		(std::cin) >> e;
		if (e!=stopflag)
		{
			tmp = new node<eT>(e);
			p->right = tmp;
			que.enQueue(tmp);
		}
	}
	
}

/* ====== Private Functions [BTree] ====== */

template <typename eT>
int BTree<eT>::Size(node<eT>* t) const
{
	if (!t) return 0;
	return Size(t->left)+Size(t->right)+1;
}

template <typename eT>
int BTree<eT>::Height(node<eT>* t) const
{
	if (!t) return 0;
	return (max(Height(t->left),Height(t->right))+1);
}

template <typename eT>
void BTree<eT>::PreOrder(node<eT> *t) const
{
	if (!t) return;
	(std::cout) << t->datum;
	PreOrder(t->left);
	PreOrder(t->right);
}

template <typename eT>
void BTree<eT>::MidOrder(node<eT> *t) const
{
	if (!t) return;
	MidOrder(t->left);
	(std::cout) << t->datum;
	MidOrder(t->right);
}

template <typename eT>
void BTree<eT>::PostOrder(node<eT> *t) const
{
	if (!t) return;
	PostOrder(t->left);
	PostOrder(t->right);
	(std::cout) << t->datum;
}

/* ====== public functions [BTree] ==== */ 

template <typename eT>
void BTree<eT>::clear()
{
	clear(root);
	root = NULL;
}

template <typename eT>
bool BTree<eT>::Empty() const
{
	return (!root);
}

template <typename eT>
node<eT>* BTree<eT>::Root() const
{
	return root;
}

template <typename eT>
int BTree<eT>::Size() const
{
	if (!root) return 0;
	
	/* === recursion version === */
	//return Size(root);
	
	/* === while version === */
	Stack<node<eT>*> sta;
	node<eT>* p;
	int count = 0;
	
	sta.push(root);
	while (!sta.empty())
	{
		p = sta.pop();
		count++;
		if (p->left)  sta.push(p->left);
		if (p->right) sta.push(p->right);
	}
	
	return count;
}

template <typename eT>
int BTree<eT>::Height() const
{
	if (!root) return 0;
	
	/* === recursion version === */
	//return Height(root);
	
	/* === while version === */
	Stack<node<eT>*> sta;
	Stack<int> sth;
	node<eT>* p;
	int h, hgh = -1;
	
	sta.push(root);
	sth.push(1);
	while (!sta.empty())
	{
		p = sta.pop();
		h = sth.pop();
		hgh = max(hgh,h);
		if (p->left)
		{
			sta.push(p->left);
			sth.push(h+1);
		}
		if (p->right)
		{
			sta.push(p->right);
			sth.push(h+1);
		}
	}
	
	return hgh;
}

template <typename eT>
void BTree<eT>::DelSubTree(node<eT>* t)
{
	clear(t);
}
	
template <typename eT>
void BTree<eT>::PreOrder() const
{
	PreOrder(root);
	
	(std::cout) << "\n"; 
}

template <typename eT>
void BTree<eT>::MidOrder() const
{
	MidOrder(root);
	
	(std::cout) << "\n";
}

template <typename eT>
void BTree<eT>::PostOrder() const
{
	/* ====== recursion version ====== */
	PostOrder(root);

	(std::cout) << "\n";
} 
	
#endif

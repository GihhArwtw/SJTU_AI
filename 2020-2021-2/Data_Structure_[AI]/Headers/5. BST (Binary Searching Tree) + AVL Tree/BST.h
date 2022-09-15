#ifndef BST_H
#define BST_H

#define NULL std::NULL

class IllegalSize_st
{
	const char* message;
	public:
		IllegalSize_st() : message("Not a legal size for the stack. Please retry.") {} 
		const char* what() { return message; }
}; 

class EmptyIssue
{
	const char* message;
	public:
		EmptyIssue() : message("Attempts to visit an empty stack detected.") {}
		const char* what() { return message; }
};

class RemoveNullPtr
{
	const char* message;
	public:
		RemoveNullPtr() : message("Attempts to delete a NULL pointer detected.") {}
		const char* what() { return message; }
};

class SizeOverflow
{
	const char* message;
	public:
		SizeOverflow(): message("Size of the BST Overflowed.") {}
		const char* what() { return message; }
};

class NotFound 
{
	const char* message;
	public:
		NotFound(): message("Failure in finding requested value.") {}
		const char* what() { return message; }
};

/* ============================== */
/* ============================== */
/* ============================== */

template <typename KEY, typename ET>
struct SET
{
	KEY key;
	ET data;
	
	SET(KEY k, ET d): key(k), data(d) {}
};

template <typename KEY, typename ET>
class node
{
	SET<KEY,ET> datum;
	node* left;          // left:  smaller
	node* right;         // right: larger
	
	node(const SET<KEY,ET>& N, node* lf = NULL, node* rg = NULL): datum(N), left(lf), right(rg) {} 
};

template <typename KEY, typename ET>
class NaiveBST
{
	node<KEY, ET>* root;
	
	void copy(const NaiveBST&);
	
	SET<KEY, ET>* search(const KEY&, node<KEY,ET>* &) const;
	void          insert(const SET<KEY,ET>&, node<KEY,ET>* &);
	void          remove(const KEY &, node<KEY,ET>* &);
	void          clear (node<KEY,ET>* &);
	
	public:
		NaiveBST();
		NaiveBST(const NaiveBST&);
		NaiveBST& operator=(const NaiveBST&);
		~NaiveBST();
		
		void          insert(const SET<KEY, ET>&);
		void          remove(const KEY&);
		SET<KEY, ET>* search(const KEY&) const;
		void          clear ();
};

#define BST NaiveBST

template <typename KEY, typename ET>
class NODE
{
	SET<KEY, ET> datum;
	int height;
	NODE* left;
	NODE* right;
	
	NODE(const SET<KEY,ET>& x, int h=0, NODE* l = NULL, NODE* r = NULL): datum(x), height(h), left(l), right(r) {}
};

class AVLTree
{
	NODE<KEY, ET>* root;
	
	 
}; 

#define AVL AVLTree
/* ============================================ */
/* ============================================ */
/* =========== STACK (embeded) ================ */
/* ============================================ */
/* ============================================ */

template <typename eT>
class Stack
{
	public:
		virtual eT   pop() = 0;
		virtual void push(const eT& x) = 0;
		virtual eT&  top() const = 0;
		virtual bool empty() const = 0;
		virtual ~Stack() {}
};

template <typename eT>
class LinkStack: public Stack<eT>
{
	struct node
	{
		eT datum;
		node* next;
		node(const eT& x, node* y=NULL) { datum = x; next = y;	}
		node(): next(NULL) {}
	};
	node* Top;
	public:
		LinkStack();
		//LinkStack(const LinkStack& x);
		//LinkStack& operator=(const LinkStack& x);
		~LinkStack();
		
		eT   pop();
		void push(const eT& x);
		bool empty() const;
};

#define Stack LinkStack

template <typename eT>
LinkStack<eT>::LinkStack()	{ Top = NULL; }

template <typename eT>
LinkStack<eT>::~LinkStack()
{
	node *p = Top, *q = NULL;
	while (p)
	{
		q = p->next;
		delete p;
		p = q;
	}
}

template <typename eT>
eT  LinkStack<eT>::pop()
{
	if (!Top) throw EmptyIssue();
	node* p = Top->next;
	eT tmp = Top->datum;
	delete Top;
	Top = p;
	return tmp;
}

template <typename eT>
void LinkStack<eT>::push(const eT& x)
{
	Top = new node(x,Top);
	//if (!Top) throw IllegalSize_st();
}

template <typename eT>
bool LinkStack<eT>::empty() const
{
	return (!Top);
}

/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ======== BST.cpp (Id return 1 exit) ======== */
/* ============================================ */
/* ============================================ */
/* ============================================ */

/* =========== Naive BST =========== */
template <typename KEY, typename ET>
NaiveBST<KEY,ET>::NaiveBST()
{
	root = NULL;
}

template <typename KEY, typename ET>
void NaiveBST<KEY,ET>::copy(const NaiveBST<KEY,ET>& x)
{
	Stack<node<KEY,ET>*> sta, stx;
	root = new node<KEY,ET> (x->root->datum);
	
	node<KEY,ET>* topa, topx;
	sta.push(root);
	stx.push(x->root);
	while (!stx.empty())
	{
		topa = sta.pop();
		topx = stx.pop();
		if (topx->left)
		{
			topa->left = new node<KEY,ET> (topx->left->datum);
			sta.push(topa->left);
			stx.push(topx->left);
		}
		if (topx->right)
		{
			topa->right = new node<KEY,ET> (topx->right->datum);
			sta.push(topa->right);
			stx.push(topx->right); 
		}
	}
}

template <typename KEY, typename ET>
void NaiveBST<KEY,ET>::clear(node <KEY,ET>* &t)
{
	if (!t) return;
	clear(t->left);
	clear(t->right);
	delete t;
}

template <typename KEY, typename ET>
NaiveBST<KEY,ET>::NaiveBST(const NaiveBST<KEY,ET>& x)	{ copy(x); }

template <typename KEY, typename ET>
NaiveBST<KEY,ET>& NaiveBST<KEY,ET>::operator=(const NaiveBST<KEY,ET>& x)  { clear(root); copy(x); }

template <typename KEY, typename ET>
NaiveBST<KEY,ET>::~NaiveBST() { clear(root); }

template <typename KEY, typename ET>
void NaiveBST<KEY,ET>::insert(const SET<KEY,ET>& x, node<KEY,ET>* &t)
{
	if (!t)  
	{
		t = new node<KEY,ET> (x);
		if (!t)  throw SizeOverflow();
	}
	if (x.key<t->datum.key)  insert(x,t->left);
	else                     insert(x,t->right);
}

template <typename KEY, typename ET>
void NaiveBST<KEY,ET>::remove(const KEY& key, node<KEY,ET>* &t)
{
	if (!t)  return; //throw RemoveNullPtr();
	if (key<t->datum.key)       remove(key,t->left);
	else if (key>t->datum.key)  remove(key,t->right);
	else
	{
		if (t->left && t->right)
		{
			node<KEY,ET>* substitute = t->right;
			while (substitute->left)  substitute = substitute->left;
			t->datum = substitute->datum;
			remove(substitute->datum.key,t->right);
		}
		else
		{
			node<KEY,ET>* tmp = t;
			t = (t->left)?(t->left):(t->right);
			delete tmp;
		}
	}
}

template <typename KEY, typename ET>
SET<KEY,ET>* NaiveBST<KEY,ET>::search(const KEY& key, node<KEY,ET>* &t) const
{
	if (!t)  throw NotFound();
	if (key<t->datum.key)  return search(key,t->left);
	if (key>t->datum.key)  return search(key,t->right);
	return t->datum;
}

template <typename KEY, typename ET>
void NaiveBST<KEY,ET>::insert(const SET<KEY,ET>& x)        { insert(x,root); }

template <typename KEY, typename ET>
void NaiveBST<KEY,ET>::remove(const KEY& x)                { remove(x,root); }

template <typename KEY, typename ET>
SET<KEY,ET>* NaiveBST<KEY,ET>::search(const KEY& x) const  { serach(x,root); }

template <typename KEY, typename ET>
void NaiveBST<KEY,ET>::clear()                             { clear(root); }

#endif

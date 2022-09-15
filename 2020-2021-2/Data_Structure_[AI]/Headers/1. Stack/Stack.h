#ifndef STACK_H
#define STACK_H
#include <iostream> 

class IllegalSize
{
	const char* message;
	public:
		IllegalSize() : message("Not a legal size. Please retry.") {} 
		const char* what() { return message; }
}; 

class EmptyIssue
{
	const char* message;
	public:
		EmptyIssue() : message("Attempts to visit an empty stack detected.") {}
		const char* what() { return message; }
};

template <typename eT>
class Stack
{
	public:
		virtual eT&  pop() = 0;
		virtual void push(const eT& x) = 0;
		virtual eT&  top() const = 0;
		virtual bool empty() const = 0;
		virtual ~Stack() {}
};

template <typename eT>
class SeqStack: public Stack<eT>
{
	eT* data;
	int Top, volume;
	void DoubleSpace();
	public:
		SeqStack();
		SeqStack(const SeqStack& x);
		SeqStack& operator=(const SeqStack& x);
		~SeqStack();
		
		eT&  pop();
		void push(const eT& x);
		eT&  top() const;
		bool empty() const;
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
		LinkStack(const LinkStack& x);
		LinkStack& operator=(const LinkStack& x);
		~LinkStack();
		
		eT&  pop();
		void push(const eT& x);
		eT&  top() const;
		bool empty() const;
};

#define Stack LinkStack

/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ========Stack.cpp (Id return 1 exit)======== */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */

/* =============== SeqStack ============== */

template <typename eT>
void SeqStack<eT>::DoubleSpace()
{
	volume *= 2;
	eT* tmp = new eT[volume];
	if (!tmp)  throw IllegalSize();
	for (int i=0; i<=Top; i++)	tmp[i] = data[i]; 
	delete []data;
	data = tmp;
}

template <typename eT>
SeqStack<eT>::SeqStack()
{
	data = new eT [10];
	Top = -1; 
	volume = 10;
}

template <typename eT>
SeqStack<eT>::SeqStack(const SeqStack<eT>& x)
{
	data = new eT[x.volume];
	for (int i=0; i<=x.Top; i++)  data[i] = x.data[i];
	Top = x.Top;	volume = x.volume;
}

template <typename eT>
SeqStack<eT>& SeqStack<eT>::operator=(const SeqStack<eT>& x)
{
	if (this==&x) return *x;
	delete []data;
	data = new eT[x.volume];
	for (int i=0; i<=x.Top; i++)  data[i] = x.data[i]; 
	Top = x.Top;	volume = x.volume;
	return *this;
}

template <typename eT>
SeqStack<eT>::~SeqStack()
{
	delete []data;
}

template <typename eT>
eT& SeqStack<eT>::pop()
{
	if (Top==-1) throw EmptyIssue();
	return data[Top--];
}

template <typename eT>
void SeqStack<eT>::push(const eT& x)
{
	if (Top+1==volume)  DoubleSpace();
	data[++Top] = x;
}

template <typename eT>
eT& SeqStack<eT>::top() const
{
	if (Top==-1)  throw EmptyIssue();
	return data[Top];
}

template <typename eT>
bool SeqStack<eT>::empty() const
{
	return (Top==-1);
}

/* =============== LinkStack ============== */

template <typename eT>
LinkStack<eT>::LinkStack()	{ Top = NULL; }

template <typename eT>
LinkStack<eT>::LinkStack(const LinkStack<eT>& x)
{
	if (!x.Top)  { Top = NULL; return;}
	Top = new node(x.Top->datum);
	node* p = x.Top->next;
	while (p)
	{
		node* tmp = new node(p->datum);
		Top->next = tmp;
		Top = tmp;
	}
}

template <typename eT>
LinkStack<eT>& LinkStack<eT>::operator=(const LinkStack<eT>& x) 
{
	if (this==&x)  return *this;
	if (!x.Top)  { Top = NULL; return *this; }
	Top = new node(x.Top->datum);
	node* p = x.Top->next;
	while (p)
	{
		node* tmp = new node(p->datum);
		Top->next = tmp;
		Top = tmp;
	}
}

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
eT& LinkStack<eT>::pop()
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
}

template <typename eT>
eT& LinkStack<eT>::top() const
{
	if (!Top) throw EmptyIssue();
	return Top->datum;
}

template <typename eT>
bool LinkStack<eT>::empty() const
{
	return (!Top);
}
	
#endif

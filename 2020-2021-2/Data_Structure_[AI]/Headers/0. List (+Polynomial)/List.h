#ifndef LIST_H
#define LIST_H
#include <iostream>
using namespace std;

class OutOfBound 
{
	const char *message;
	public:
		OutOfBound(): message("Out of bound. PLease retry") {};
		const char* what() const {return message;}
};

class IllegalSize 
{
	const char *message;
	public:
		IllegalSize(): message("Not a legal size. PLease retry") {};
		const char* what() const {return message;}
};

template <typename eT>
class List
{
	public:
		virtual int length() const = 0;
		virtual int search(const eT& x) const = 0;
		virtual eT& visit(int i) const = 0;
		virtual void insert(int i, const eT& x) = 0;
		virtual void remove(int i) = 0;
		virtual void clear() = 0;
		virtual void traverse() const = 0;
		virtual ~List() {};
};

//index of all elements in List: [0..n-1]

template <typename eT>
class SeqList: public List<eT>
{
	eT* data;
	int size;
	int volume;	
	void doubleSpace();
	public:
		SeqList(int N=10);
		SeqList(const SeqList<eT>& x);
		SeqList& operator=(const SeqList& x);
		~SeqList();
		
		int length() const;
		int search(const eT& x) const;
		eT& visit(int i) const;
		void insert(int i, const eT&x);
		void remove(int i);
		void clear();
		void traverse() const;
		
		friend ostream& operator<<(ostream& os, const SeqList<eT>& x)
		{	x.traverse();	return os; }
};

/*
SeqList:
List Using Array 
*/

template <typename eT>
class SgLinkList: public List<eT>
{
	struct node
	{
		eT datum;
		node* next;
		node(const eT& x, node* y=NULL):datum(x),next(y) {}
		node():next(NULL) {}
	};
	node* head;
	public:
		SgLinkList();
		SgLinkList(const SgLinkList<eT>& x);
		SgLinkList& operator=(const SgLinkList& x);
		~SgLinkList();
		
		int length() const;
		int search(const eT& x) const;
		eT& visit(int i) const;
		void insert(int i, const eT&x);
		void remove(int i);
		void clear();
		void traverse() const;
		friend ostream& operator<<(ostream& os, const SgLinkList<eT>& x)
		{	x.traverse();	return os; }
};

#define LinkList SgLinkList

/*
SgLinkList:
List Using Single Link 
*/
 
template <typename eT>
class DbLinkList: public List<eT>
{
	struct node
	{
		eT datum;
		node* prev;
		node* next;
		node(const eT& x, node* y=NULL, node* z=NULL) { datum = x; prev = y; next = z;}
		node():prev(NULL), next(NULL) {}
	};
	node* head;
	node* tail;
	public:
		DbLinkList();
		DbLinkList(const DbLinkList& x);
		DbLinkList& operator=(const DbLinkList& x);
		~DbLinkList();
		
		int length() const;
		int search(const eT& x) const;
		eT& visit(int i) const;
		void insert(int i, const eT&x);
		void remove(int i);
		void clear();
		void traverse() const;
		friend ostream& operator<<(ostream& os, const DbLinkList<eT>& x)
		{	x.traverse();	return os; }; 
};

/*
DbLinkList:
List Using Double Link
*/

/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ====== List.cpp (Id return 1 exit) ======= */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */
/* ========================================== */

//#include <iostream>
//#include "List.h"

/* =============== SeqList =============== */
template <typename eT>
void SeqList<eT>::doubleSpace()
{
	eT* tmp = data;
	volume *= 2;
	data = new eT[volume];
	for (int i=0; i<size; i++) data[i] = tmp[i];
	delete []tmp;	
}

template <typename eT>
SeqList<eT>::SeqList(int n)
{
	data = new eT[n];
	if (!data) throw IllegalSize();
	size = 0;
	volume = n; 
}

template <typename eT>
SeqList<eT>::SeqList(const SeqList<eT>& x)
{
	data = new eT[x.volume];
	for (int i=0; i<x.size; i++)	data[i] = x.data[i];
	volume = x.volume;
	size = x.size;
}

template <typename eT>
SeqList<eT>& SeqList<eT>::operator=(const SeqList<eT>& x)
{
	if (this==&x)  return *this;
	delete []data;
	data = new eT[x.volume];
	for (int i=0; i<x.size; i++)	data[i] = x.data[i];
	volume = x.volume;
	size = x.size;
	return *this;
}

template <typename eT>
int SeqList<eT>::length() const
{
	return size;
}

template <typename eT>
int SeqList<eT>::search(const eT& x) const
{
	for (int j=0; j<size; j++)
		if (data[j] == x) return j;
	return -1;
}

template <typename eT>
eT& SeqList<eT>::visit(int i) const
{
	if (i<0 || i>=size)  throw OutOfBound();
	return data[i];
}

template <typename eT>
void SeqList<eT>::insert(int i, const eT& x)
{
	if (i<0 || i>size)  throw OutOfBound();
	if (size==volume)  doubleSpace();
	for (int j=size-1; j>=i; j--)  data[j+1] = data[j];
	data[i] = x;
	size++;
}

template <typename eT>
void SeqList<eT>::remove(int i)
{
	if (i<0 || i>=size)  throw OutOfBound();
	size--;
	for (int j=i; j<size; j++)  data[j] = data[j+1];
}

template <typename eT>
void SeqList<eT>::clear()
{
	size = 0;
}

template <typename eT>
void SeqList<eT>::traverse() const
{
	for (int j=0; j<size; j++)  (std::cout) << data[j] << " ";
	(std::cout) << "[END]\n";
}

template <typename eT>
SeqList<eT>::~SeqList()
{
	delete []data;
}

/* =============== SgLinkList =============== */

template <typename eT>
SgLinkList<eT>::SgLinkList()
{
	head = new node();
}
	
template <typename eT>
SgLinkList<eT>::SgLinkList(const SgLinkList<eT>& x)
{
	head = new node();
	node* p = x.head->next;
	int i = 0;
	while (p)
	{
		insert(i,p->datum);
		p = p->next;
		i++;
	}
}

template <typename eT>
SgLinkList<eT>& SgLinkList<eT>::operator=(const SgLinkList<eT>& x)
{
	if (this==&x)  return *this;
	clear();
	node* p = x.head->next;
	int i = 0;
	while (p)
	{
		insert(i,p->datum);
		p = p->next;
		i++;
	}
	return *this;
}

template <typename eT>
void SgLinkList<eT>::clear()
{
	node* p = head->next;
	while (p)
	{
		delete head;
		head = p;
		p = head->next;
	}
	head = new node();
}	

template <typename eT>
SgLinkList<eT>::~SgLinkList()
{
	node* p = head->next;
	while (p)
	{
		delete head;
		head = p;
		p = head->next;
	}
}

template <typename eT>
int SgLinkList<eT>::length() const
{
	node* p = head->next;
	int len = 0;
	while (p)
	{
		p = p->next;
		len++;
	}
	return len;
}

template <typename eT>
int SgLinkList<eT>::search(const eT& x) const
{
	node* p = head->next;
	int i = 0;
	while (p)
	{
		if (p->datum==x) return i;
		p = p->next;
		i++;
	}
	return -1;
}
	
template <typename eT>
eT& SgLinkList<eT>::visit(int i) const
{
	if (i<0) throw OutOfBound();
	node* p = head->next;
	int j = 0;
	while (p)
	{
		if (i==j) return p->datum;
		p = p->next;
		i++;
	}
	throw OutOfBound();
}


template <typename eT>
void SgLinkList<eT>::insert(int i, const eT& x)
{
	if (i<0) throw OutOfBound();
	node* p = head;
	int j = 0;
	while (p)
	{
		if (i==j)
		{
			node* tmp = new node(x,p->next); 
			p->next = tmp;
			return;
		}
		j++;
		p = p->next;
	}
	throw OutOfBound();
}
	
template <typename eT>
void SgLinkList<eT>::remove(int i)
{
	if (i<0) throw OutOfBound();
	node* p = head, *tmp;
	int j = 0;
	while (p)
	{
		if (i==j)
		{
			tmp = p->next;
			p->next = tmp->next;
			delete tmp;
			return;
		}
		j++;
		p = p->next;
	}
	throw OutOfBound();
}
	
template <typename eT>
void SgLinkList<eT>::traverse() const
{
	node* p = head->next;
	while (p)
	{
		(std::cout) << p->datum << " ";
		p = p->next;	
	}
	(std::cout) << "[END]\n";
}

/* =============== DbLinkList =============== */

template <typename eT>
DbLinkList<eT>::DbLinkList()
{
	head = new node();	tail = new node();
	head->next = tail;	tail->prev = head;
}

template <typename eT>
DbLinkList<eT>::DbLinkList(const DbLinkList& x)
{
	head = new node();	tail = new node();
	head->next = tail;	tail->prev = head;
	node *p = x.head->next;
	int i = 0;
	while (p!=x.tail)
	{
		insert(i,p->datum);
		p = p->next;
		i++;
	}
}

template <typename eT>
DbLinkList<eT>& DbLinkList<eT>::operator=(const DbLinkList& x)
{
	if (this==&x) return *this;
	clear();
	node *p = x.head->next;
	int i = 0;
	while (p!=x.tail)
	{
		insert(i,p->datum);
		p = p->next;
		i++;
	}
	return *this;
}

template <typename eT>
void DbLinkList<eT>::clear()
{
	node* p = NULL, *q = tail;
	while (q)
	{
		p = q->prev;
		delete q;
		q = p;
	}
	head = new node();
	tail = new node();
	head->next = tail;
	tail->prev = head; 
}

template <typename eT>
DbLinkList<eT>::~DbLinkList()
{
	node* p = NULL;
	while (tail)
	{
		p = tail->prev;
		delete tail;
		tail = p;
	}
}

template <typename eT>
int DbLinkList<eT>::length() const
{
	node *p = head->next;
	int i = 0;
	while (p!=tail)
	{
		i++;
		p = p->next;
	}
	return i;
}

template <typename eT>
int DbLinkList<eT>::search(const eT& x) const
{
	node *p = head->next;
	int i = 0;
	while (p!=tail)
	{
		if (p->datum == x) return i;
		p = p->next;
		i++;
	}
	
	return -1;
} 
	
template <typename eT>
eT& DbLinkList<eT>::visit(int i) const
{
	if (i<0) throw OutOfBound();
	node *p = head->next;
	int j = 0;
	while (p!=tail)
	{
		if (j==i) return p->datum;
		p = p->next;
		j++;
	}
	throw OutOfBound(); 
}

template <typename eT>
void DbLinkList<eT>::insert(int i, const eT& x)
{
	if (i<0) throw OutOfBound();
	node *p = head;
	int j = 0;
	while (p!=tail)
	{
		if (j==i)
		{
			node* tmp = new node(x, p, p->next);
			p->next = tmp;
			tmp->next->prev = tmp;
			return;
		}
		p = p->next;
		j++;
	}
	throw OutOfBound();
}

template <typename eT>
void DbLinkList<eT>::remove(int i)
{
	if (i<0) throw OutOfBound();
	node* p = head;
	int j = 0;
	while (p->next!=tail)
	{
		if (j==i)
		{
			node* q = p->next->next;
			delete p->next;
			p->next = q;
			q->prev = p;
			return;
		}
		j++;
		p = p->next;
	}
	throw OutOfBound();
} 

template <typename eT>
void DbLinkList<eT>::traverse() const
{
	node* p = head->next;
	while (p!=tail)
	{
		(std::cout) << p->datum << " ";
		p = p->next;		
	}
	(std::cout) << "[END]\n";
}


#endif

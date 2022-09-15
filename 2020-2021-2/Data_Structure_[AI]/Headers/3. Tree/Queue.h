#ifndef QUEUE_H
#define QUEUE_H

#include <iostream>

class IllegalSize
{
	const char* message;
	public:
		IllegalSize(): message("Illegal Size. Please retry.") {	}
		const char* what() { return message;	}
};

class OutOfBound
{
	const char* message;
	public:
		OutOfBound(): message("Out of bound. Please retry.") {	}
		const char* what() { return message;	}
};

template <typename eT>
class Queue
{
	public:
		virtual void enQueue(const eT&) = 0;
		virtual eT   deQueue() = 0;
		virtual bool empty() const = 0;
		virtual eT   front() const = 0;
		virtual ~Queue() {};
};

template<typename eT>
class SeqQueue: public Queue<eT>
{
	eT* data;
	int head;
	int tail;
	int length;
	
	void doubleSpace();
	
	public:
		SeqQueue(int=10); 
		SeqQueue(const SeqQueue<eT>&);
		SeqQueue<eT>& operator=(const SeqQueue<eT>&);
		~SeqQueue();
	
		void enQueue(const eT&);
		eT   deQueue();
		eT   front() const;
		bool empty() const;
};


template<typename eT>
class LinkQueue: public Queue<eT>
{
	struct node
	{
		eT datum;
		node* next;
		node(const eT& x, node* y = NULL) { datum = x; next = y;	} 
		node(): next(NULL) {}
	};
	
	node* head;
	node* tail;
	
	
	public:
		LinkQueue();
		LinkQueue(const LinkQueue<eT>&);
		LinkQueue<eT>& operator=(const LinkQueue<eT>&); 
		~LinkQueue();
	
		void enQueue(const eT&);
		eT   deQueue();
		bool empty() const;
		eT   front() const;
		
};

#define Queue LinkQueue

/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ========Queue.cpp (Id return 1 exit)======== */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ============================================ */

/* =============== SeqQueue ============== */

template<typename eT>
SeqQueue<eT>::SeqQueue(int ini)
{
	data = new int[ini];
	if (!data)  throw IllegalSize();
	head = 0; tail = 0; length = ini;
}

template<typename eT>
SeqQueue<eT>::SeqQueue(const SeqQueue<eT>& x)
{
	eT* tmp = new int[x.length];
	for (int i=1; i<=length; i++)  tmp[i] = x.data[(i+head)%length];
	delete []data;
	data = tmp;
	head = 0; tail = x.length-1;  length = x.length;
} 

template<typename eT>
SeqQueue<eT>& SeqQueue<eT>::operator=(const SeqQueue<eT>& x)
{
	if (this==&x)  return *this;
	
	eT* tmp = new int[x.length];
	for (int i=1; i<=length; i++)  tmp[i] = x.data[(i+head)%length];
	delete []data;
	data = tmp;
	head = 0; tail = x.length-1;  length = x.length;
	
	return *this; 
}

template<typename eT>
SeqQueue<eT>::~SeqQueue()
{
	delete []data;
}

template<typename eT>
void SeqQueue<eT>::doubleSpace()
{
	eT* tmp = new eT[length*2];
	if (!tmp) throw IllegalSize();
	
	for (int i=1; i<length; i++)  tmp[i] = data[(i+head)%length];
	delete []data;
	data = tmp;
	head = 0; tail = length-1;
	length *= 2;
}

template<typename eT>
void SeqQueue<eT>::enQueue(const eT& x)
{
	if ((tail+1)%length == head)  doubleSpace();
	data[tail=(tail+1)%length] = x;
}

template<typename eT>
eT SeqQueue<eT>::deQueue()
{
	if (empty()) throw OutOfBound();
	head = (head+1)%length;
	eT tmp = data[head];
	return tmp;
}

template<typename eT>
eT SeqQueue<eT>::front() const
{
	if (empty()) throw OutOfBound();
	eT tmp = data[head+1];
	return tmp;
}

template<typename eT>
bool SeqQueue<eT>::empty() const
{
	return (head==tail);
}

/* =============== LinkQueue ============== */
template<typename eT>
LinkQueue<eT>::LinkQueue()
{
	head = tail = NULL;
}

template<typename eT>
LinkQueue<eT>::LinkQueue(const LinkQueue<eT>& x)
{
	if (!x.head) return;
	tail = head = new node(x.head->datum);
	node* p = x.head->next;
	while (p)
	{
		node* tmp = new node(p->datum);
		tail->next = tmp;
		tail = tail->next;
		p = p->next;
	}
}

template<typename eT>
LinkQueue<eT>& LinkQueue<eT>::operator=(const LinkQueue<eT>& x)
{
	if (this == &x) return *this;
	node* p = head, *q = NULL;
	while (p)
	{
		q = p->next;
		delete p;
		p = q;
	}
	
	if (!x.head)
	{
		head = tail = NULL; return *this;
	}
	tail = head = new node(x.head->datum);
	p = x.head->next;
	while (p)
	{
		node* tmp = new node(p->datum);
		tail = (tail->next = tmp);
		p = p->next;
	}
	
	return *this;
}

template<typename eT>
LinkQueue<eT>::~LinkQueue()
{
	node* p = head, *q = NULL;
	while (p)
	{
		q = p->next;
		delete p;
		p = q;
	}
}

template<typename eT>
void LinkQueue<eT>::enQueue(const eT& x)
{
	node* tmp = new node(x);
	if (!head)
	{
		head = tail = tmp;
		return;
	}
	tail = (tail->next = tmp);
}

template<typename eT>
eT LinkQueue<eT>::deQueue()
{
	if (empty()) throw OutOfBound();
	eT tmp = head->datum;
	node* p = head;
	head = head->next;
	delete p;
	return tmp;
}

template<typename eT>
eT LinkQueue<eT>::front() const
{
	if (empty()) throw OutOfBound();
	eT tmp = head->datum;
	return tmp;
}

template<typename eT>
bool LinkQueue<eT>::empty() const
{
	return (head==NULL);
}


#endif

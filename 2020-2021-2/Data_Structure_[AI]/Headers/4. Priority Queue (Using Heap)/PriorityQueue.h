#ifndef PRIORITYQUEUE_H
#define PRIORITYQUEUE_H
#include <algorithm>

class IllegalSize
{
	const char* message;
	public:
		IllegalSize(): message("Illegal Size Detecteed. PLease retry.\n") {}
		const char* what() { return message; }
};

class IllegalVisit 
{
	const char* message;
	public:
		IllegalVisit(): message("Visiting an empty queue. PLease retry.\n") {}
		const char* what() { return message; }
};

template <typename eT>
class PriorityQueue
{
	int size;
	int volume;
	eT* data;
	bool cmp;   // cmp = 1: Superiority(MAX) Queue; cmp = 0 : Inferiority(MIN) Queue.
	//bool (*cmp)(const eT&, const eT&);
		
	void DoubleSpace(); 	
	
	public:
		PriorityQueue(int=10, bool=true);
		//PriorityQueue(int=10, bool (*b)(const eT&, const eT&) = std::max);
		PriorityQueue(const PriorityQueue&);
		PriorityQueue& operator=(const PriorityQueue&);
		~PriorityQueue();
		
		void enQueue(const eT&);
		eT&  vsQueue() const;     // visit the root
		eT   deQueue();
		bool empty() const;
};

#define PriQue PriorityQueue

/* ============================================ */
/* ============================================ */
/* ============================================ */
/* === PriorityQueue.cpp (Id return 1 exit) === */
/* ============================================ */
/* ============================================ */
/* ============================================ */

template <typename eT>
PriorityQueue<eT>::PriorityQueue(int x, bool b)//bool (*b)(const eT&, const eT&)) 
{
	if (x<=0) throw IllegalSize();
	size = 0;
	volume = x;
	
	cmp = b;
}

template <typename eT>
PriorityQueue<eT>::PriorityQueue(const PriorityQueue& x)
{
	volume = x.volume;
	size = x.size;
	cmp = x.cmp;
	for (int i=1; i<=size; i++)	data[i] = x.data[i];	
}

template <typename eT>
PriorityQueue<eT>& PriorityQueue<eT>::operator=(const PriorityQueue<eT>& x)
{
	if (&this==x)  return;
	volume = x.volume;
	size = x.size;
	cmp = x.cmp;
	for (int i=1; i<=size; i++)	data[i] = x.data[i];
}

template <typename eT>
void PriorityQueue<eT>::DoubleSpace()
{
	volume *= 2;
	eT* tmp = new eT [volume];
	for (int i=0; i<volume/2; i++)	tmp[i] = data[i];
	delete []data;
	data = tmp;
}

template <typename eT>
PriorityQueue<eT>::~PriorityQueue()
{
	delete []data;
}

template <typename eT>
bool PriorityQueue<eT>::empty() const
{
	return (!size);
}

template <typename eT>
void PriorityQueue<eT>::enQueue(const eT& x)
{
	if (size==volume-1) DoubleSpace();
	data[++size] = x;
	int i=size;
	while (i/2 && cmp-(data[i/2]>=x)) //x==cmp(data[i/2],x)) 
	{
		data[i] = data[i/2];
		i = i/2;
	}
	data[i] = x;
}

template <typename eT>
eT&  PriorityQueue<eT>::vsQueue() const
{
	if (empty()) throw IllegalVisit();
	return data[1];
}

template <typename eT>
eT   PriorityQueue<eT>::deQueue()
{
	if (empty()) throw IllegalVisit();
	eT tmp = data[1];
	eT x = (data[1] = data[size--]);
	int i = 1;
	while (i*2<=size)
	{
		int child = i*2;
		if (i*2+1<=size && cmp-(data[i*2+1]<=data[i*2]))/*data[i*2+1]==cmp(data[i*2+1],data[i*2]))*/
		  child++;
		if (cmp-(data[child]>data[i])) break;
		data[i] = data[child];
	}
	data[i] = x;
	return tmp;
}

#endif

#include <cstdio>
using namespace std;

class OutOfBound
{
	const char* message;
	public:
		OutOfBound() : message("Out Of Bound. Please retey.") {}
}; 

template <typename eT>
class CyQueue
{
	public:
	eT* data;
	int head, tail, len;
	
	public:
	CyQueue(int=10);
	CyQueue(const CyQueue& x);
	CyQueue& operator=(const CyQueue& x);
	~CyQueue();
	
	void enQueue(const eT& x);
	void deQueue();
	eT& getHead() const;
	bool empty() const;
};

template <typename eT>
CyQueue<eT>::CyQueue(int ini)
{
	data = new eT[ini];
	head = 0; tail = 0; len = ini;
}

template <typename eT>
CyQueue<eT>::CyQueue(const CyQueue& x)
{
	data = new eT[x.len];
	for (int i=0; i<x.len; i++)  data[i] = x.data[i];
	head = x.head; tail = x.tail; len = x.len;
} 

template <typename eT>
CyQueue<eT>& CyQueue<eT>::operator=(const CyQueue& x)
{
	delete []data;
	data = new eT[x.len];
	for (int i=0; i<x.len; i++)  data[i] = x.data[i];
	head = x.head; tail = x.tail; len = x.len;
} 

template <typename eT>
CyQueue<eT>::~CyQueue()
{
	delete []data; 
}

template <typename eT>
void CyQueue<eT>::enQueue(const eT& x)
{
	tail = (tail+1)%len;
	data[tail] = x;
	printf("%d %d\n",tail,(tail-head-1)%len+1);
}

template <typename eT>
void CyQueue<eT>::deQueue()
{
	if (!empty())  head = (head+1)%len;
	printf("%d %d\n",head,(tail-head-1)%len+1);
}

template <typename eT>
bool CyQueue<eT>::empty() const
{
	return (head==tail);
}

template <typename eT>
eT& CyQueue<eT>::getHead() const
{
	if (empty()) throw OutOfBound();
	return data[head];
}

int main()
{
	int s,n;
	scanf("%d %d",&s,&n);
	CyQueue<int> a(s+1);
	
	int op, x;
	for (int i=0;i<n;i++)
	{
		scanf("%d",&op);
		if (op)		{ a.deQueue(); }
		else	{ scanf("%d",&x); a.enQueue(x); }
	//	printf("%d %d\n",a.head,a.tail);
	}
	
	return 0;
}

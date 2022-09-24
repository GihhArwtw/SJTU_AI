#include <cstdio>
using namespace std; 
 
class SeqStack
{
	long long *data, *min;
	long long Top;
	
	public:	
	SeqStack() { Top = -1; data = new long long[1000050]; min = new long long[1000050];}
	void push(long long x); 
	bool pop();
	long long top() const;
	long long getMin() const;
	~SeqStack() {}
}; 
 
void SeqStack::push(long long x)
{
	data[++Top] = x;
	if (Top==0) { min[Top] = x; return; }
	min[Top] = min[Top-1];
	if (min[Top]>x)  min[Top] = x;
}
 
bool SeqStack::pop()
{
	if (Top==-1) return true;
	Top--; return false;
}
 
long long SeqStack::top() const
{
	if (Top==-1) return -1;
	return data[Top];
}
 
long long SeqStack::getMin() const
{
	if (Top==-1) return -1;
	return min[Top];
}
 
int main()
{
	int n;
	scanf("%lld",&n);
	
	long long x, y;
	SeqStack a;
	for (int i=0; i<n; i++)
	{
		scanf("%lld",&x);
		switch (x)
		{
			case 0: scanf("%lld",&y);
					a.push(y);
					break;
			case 1: if (a.pop())  printf("Empty\n");
					break;
			case 2: y = a.top();
					if (y>=0)  { printf("%lld\n",y); break; }
					printf("Empty\n");
					break;
			case 3: y = a.getMin();
					if (y>=0) { printf("%lld\n",y); break; }
					printf("Empty\n");
					break;
		}
	}
	
	return 0;
}

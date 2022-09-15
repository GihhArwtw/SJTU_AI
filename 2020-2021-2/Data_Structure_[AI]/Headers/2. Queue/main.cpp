#include <iostream>
#include "Queue.h"
#include <cstdio>
/* run this program using the console pauser or add your own getch, system("pause") or input loop */

#include <iostream>

int main(int argc, char** argv) 
{

	int s,n;
	scanf("%d %d",&s,&n);
	LinkQueue<int> a;//SeqQueue<int> a(s+1);
	
	int op, x;
	for (int i=0;i<n;i++)
	{
		scanf("%d",&op);
		if (op)		{ printf("%d\n",a.deQueue()); }
		else	{ scanf("%d",&x); a.enQueue(x); printf("%d\n",a.front()); }
	//	printf("%d %d\n",a.head,a.tail);
	}
	
	return 0;
}

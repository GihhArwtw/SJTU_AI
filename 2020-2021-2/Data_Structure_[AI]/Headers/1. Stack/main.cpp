#include <cstdio>
#include "Stack.h"

int main()
{
	SeqStack<int> a;
	
	int n, op, num;
	scanf("%d",&n);
	
	for (int i=0; i<n; i++)
	{
		scanf("%d %d",&op,&num);
		if (op==1)
		{
			printf("OK\n");
			a.push(num);
			continue; 
		}
		if (a.empty())  { printf("ERROR\n"); continue;	}
		if (a.pop()==num)  printf("YES\n");
		else printf("NO\n");
	}
	
	return 0;
}

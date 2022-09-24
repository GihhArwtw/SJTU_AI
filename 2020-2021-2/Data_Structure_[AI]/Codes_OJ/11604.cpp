#include <cstdio>

#define MOD 1000000007

int total = 0;

int search(int wait, int in, int top, int cost, int all)
{
	if (wait==0) 
	{
		
	}
	int tmp = search(wait-1,in+1,wait,,all);
	if (in>0) tmp += search(wait,in-1,(cost-top)%MOD);
	return tmp;
}

int main()
{
	int n;
	scanf("%d",&n);
	printf("%d",search(n,0),0,0); 
	return 0;
}

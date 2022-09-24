#include <cstdio>
using namespace std;

int main()
{
	int n;
	scanf("%d",&n);
	
	int time=0, wait=0, x, y;
	for (int i=0; i<n; i++)
	{
		scanf("%d %d",&x,&y);
		if (x>=time)	{ time = x+y; continue; }
		wait += time-x;
		time += y;
	}
	
	double ave = wait/double(n);
	
	printf("%.2f",ave);
	return 0;
}

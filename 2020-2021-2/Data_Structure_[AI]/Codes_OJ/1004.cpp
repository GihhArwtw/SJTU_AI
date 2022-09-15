#include <iostream>

using namespace std;

int main()
{
	int n,time,up,fwd,dn;
	cin >> time >> n >> up >> fwd >> dn;
	
	int tmp = 0, now = 0, step = 0, i;
	char ch,ws;
	for (i=0; i<n; i++)
	{
		cin >> ch;
		switch (ch)
		{
			case 'f': tmp = fwd*2; break;
			case 'u': case'd': tmp = (up+dn); break;
		}
		if (now+tmp>time)  break;
		now += tmp;
		step ++;
	}
	
	cout << step;
	
	return 0;
	
} 

#include <iostream>
using namespace std;

int main()
{
	int hgh, h, n;
	cin >> hgh >> h >> n;
	hgh += h;
	int total = 0, x;
	for (int i=0; i<n; i++)
	{
		cin >> x;
		if (x<=hgh)  total++;
	}
	cout << total;
	
	return 0;
}

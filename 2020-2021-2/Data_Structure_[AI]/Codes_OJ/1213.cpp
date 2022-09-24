#include <iostream>

using namespace std; 

int main()
{
	int xdrct[8] = {1,1,1,0,-1,-1,-1,0}, ydrct[8] = {-1,0,1,1,1,0,-1,-1};
	
	int n,m;
	cin >> n >> m;
	
	short** a = new short*[n];
	char ch = ' ';
	while (ch!='.' && ch!='W')  ch = getchar();
	for (int i=0; i<n; i++)
	{
		a[i] = new short[m];
		for (int j=0; j<m; j++)
		{
			if (ch=='W')  a[i][j]=1; else a[i][j]=0;
			ch = getchar();
		}
		while ((i<n-1) && ch!='.' && ch!='W')  ch = getchar();
	}
	
	int ponds = 0;
	int head = 0, tail = 0, xx, yy;
	int* qux = new int[n*m+50], *quy = new int[n*m+50];
	for (int i=0; i<n; i++)
	{
		for (int j=0; j<m; j++)
		if (a[i][j])
		{
			ponds++;
			a[i][j] = 0;
	
			head = tail = 0;
			qux[head] = i;  quy[head] = j;
			while (head<=tail)
			{
				for (int i=0; i<8; i++)
				{
					xx = qux[head] + xdrct[i];
					yy = quy[head] + ydrct[i];
					if (xx>=0 && xx<n && yy>=0 && yy<m && a[xx][yy])
					{
						a[xx][yy] = 0; tail++; qux[tail] = xx; quy[tail] = yy;
					}
				}
				head++;
			}
		}
	}
	
	cout << ponds;
	delete []qux;
	delete []quy;
	
	for (int i=0; i<n; i++)  delete []a[i];
	delete []a;
	
	return 0;
}

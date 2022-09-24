#include<iostream>

using namespace std; 

void expand(int x, int y, int n, int m, short** &a)
{
	static int xdrct[8] = {1,1,1,0,-1,-1,-1,0}, ydrct[8] = {-1,0,1,1,1,0,-1,-1};
							// direction_x, direction_y
	
	a[x][y] = 0;
	
	int head = 0, tail = 0;
	int* qux = new int[n*m+50], *quy = new int[n*m+50];
	qux[head] = x;  quy[head] = y;
	int xx,yy;
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
	
	delete []qux;
	delete []quy;
		
}

int main()
{
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
	for (int i=0; i<n; i++)
	{
		for (int j=0; j<n; j++)
		if (a[i][j])
		{
			ponds++;
			expand(i,j,n,m,a);
		}
	}
	
	cout << ponds;
	
	for (int i=0; i<n; i++)  delete []a[i];
	delete []a;
	
	return 0;
}

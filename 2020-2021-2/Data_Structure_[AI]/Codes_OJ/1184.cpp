#include <cstdio>
using namespace std;

int main()
{
	//freopen("1184.txt","r",stdin);
	
	int n,m;
	scanf("%d %d",&n,&m);
	
	int* a = new int[n+5];
	for (int i=0; i<n; i++)  scanf("%d",&a[i]);
	
	int filemax = -1, filecnt = 0, remaining = n;
	int*  hol  = new int[m+5];     // heap or list
	bool* flag = new bool[m+5];    // TRUE: heap; FALSE: list.
	int   heap = -1;               // SIZE(heap)
	
	for (int i=1; i<=m; i++)  hol[i] = a[i-1];
	int scan = m-1;
	
	while (true)
	{
		filecnt++; filemax = -1;                // create a file and open it
		for (int i=1; i<=m; i++)
		{
			flag[i] = true;
			int child = i, tmp = hol[i];
			while (child>1 && tmp<hol[child/2])
			{
				hol[child] = hol[child/2];
				child = child/2;
			}
			hol[child] = tmp;
		}                                       // create the heap
		
		heap = m;
		while (heap)
		{
			filemax = hol[1];
			hol[1]  = hol[heap];
			heap--;  remaining--;                             // output into the file
			
			int par = 1;
			while (par*2<=heap)
			{
				int child = par*2;
				if (par*2+1<=heap && hol[child]>hol[par*2+1])   child++;
				if (hol[child]>=hol[par]) break;
				hol[par] = hol[child];   hol[child] = hol[heap+1];
				par = child;
			}                                                 // down-maintain the heap
			
			if (scan==n-1)   { remaining -= heap; break; }
			if (a[++scan]<filemax)
			{
				hol[heap+1] = a[scan];   flag[heap+1] = false;   // turn to list
			}
			else
			{
				hol[++heap] = a[scan];
				int child = heap, tmp = hol[heap];
				while (child>1 && tmp<hol[child/2])
				{
					hol[child] = hol[child/2];
					child = child/2;
				}
				hol[child] = tmp;
				                                                 // insert into heap
			}
		} //END while(heap) 
		
		if (!remaining)  break;
		if (scan==n-1)   {  filecnt++;  break;  }
		
		
	} // END while(true) 
	
	printf("%d",filecnt);
	
	//fclose(stdin);
	
	return 0;
}

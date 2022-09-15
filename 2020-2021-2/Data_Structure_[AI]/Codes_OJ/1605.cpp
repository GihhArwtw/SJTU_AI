#include <cstdio>
#include <fstream>
using namespace std;

inline bool Right(char ch)
{
	return (ch==')' || ch==']' || ch=='}');
}

inline char Pair(char ch)
{
	if (ch==')') return '(';
	if (ch==']') return '[';
	if (ch=='}') return '{';
	return ' ';
} 
 
int main()
{	
	int n;
	scanf("%d",&n);
	
	int x, top = -1, tops = -1;
	char a[100020], stack[100020];
	bool off[100020];
	for (int i=0; i<100020; i++)  off[i] = false;
	char ch;
	for (int i=0; i<n; i++)
	{
		scanf("%d",&x); 
		switch (x)
		{
			case 1: ch = getchar(); ch = getchar();	a[++top] = ch;
					if (tops==-1)  { stack[++tops] = ch; break;}
					if (stack[tops]==Pair(ch)) { tops--; off[top] = true; break;}
					stack[++tops] = ch; off[top] = false;
					break;
			case 2: if (top<0) break; top--;
					if (off[top+1])
						{ stack[++tops] = Pair(a[top+1]); break;}
					tops--;
					break;
			case 3: if (top<0) break; printf("%c\n",a[top]); break;
			case 4: if (tops==-1) { printf("YES\n"); break;}
					printf("NO\n"); break;
		}
	}
	return 0;
	
}

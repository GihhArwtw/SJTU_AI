#include <iostream>
#include "list.h"
//#include "list.cpp" 
//using namespace std;

int main()
{
	int n,x;
	cin >> n;
	
	DbLinkList<int> a, b;
	
	for (int i=0; i<n; i++)
	{
		cin >> x;
		a.insert(i,x);
	}
	
	b = a;
	DbLinkList<int> c = a;
	
	cout << "length:" << a.length() << "\n";
	//
	cout << a; //	a.traverse();
	
	cin >> x;
	cout << a.search(x) << "\n";
	
	x = a.visit(0);
	a.remove(0);
	a.insert(int(n/2),x);
	
	a.traverse();
	
	
	a.clear();
	a.traverse();
	cout << b;
	cout << c;
	
	for (int i=0; i<n; i++)
	{
		cin >> x;
		a.insert(i,x);
	}
	
	cout << "length:" << a.length() << "\n";
	
	cin >> x;
	cout << a.search(x) << "\n";

	x = a.visit(0);
	a.remove(0);
	a.insert(int(n/2),x);
	
	a.traverse();
	
	a.clear();
	
	
	return 0;
}

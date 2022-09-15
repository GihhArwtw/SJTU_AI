#include <iostream>
#include "Polynomial.h"
#include <iostream>

using namespace std;

/* run this program using the console pauser or add your own getch, system("pause") or input loop */

int main(int argc, char** argv) 
{
	Polynomial<double> a,b;
	int n;
	cin >> n;
	for (int i=0; i<n; i++)
	{
		double x; int y;
		cin >> x >> y;
		a.addterm(y,x);
	}
	
	cin >> n;
	for (int i=0; i<n; i++)
	{
		double x; int y;
		cin >> x >> y;
		b.addterm(y,x);
	}
	
	Polynomial<double> c = a, d;
	d = b;
	c.print();
	d.print();
	
	a.print();
	b.print();
	
	a.add(a,b);
	a.print();
	b.clear();
	b.print();
	double x;
	cin >> x;
	cout << a.value(x);
	
	return 0;
}

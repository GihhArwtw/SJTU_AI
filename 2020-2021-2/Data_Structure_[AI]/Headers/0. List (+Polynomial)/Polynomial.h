#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <iostream>
using namespace std;

class IllegalPower
{
	const char* message;
	public:
		IllegalPower(): message("Negative Power Detected.") {}
		const char* what() const { return message; }
};

template <typename eT>
class Polynomial
{
	struct term
	{
		int pow;
		eT 	coe;   //coefficient
		term* next;
		term(): next(NULL) {}
		term(int x, eT y, term* z=NULL): pow(x), coe(y), next(z) {}
	};
	
	term* head;
	
	void print(term* p) const;
	void clear(int);
	
	public:
		Polynomial();
		Polynomial(const Polynomial& x);
		Polynomial& operator=(const Polynomial& x);
		~Polynomial();
		
		void clear();
		void addterm(int, eT);
		void add(const Polynomial& x, const Polynomial& y);
		void print() const;
		eT   value(eT x) const;
};



/* ========================================== */
/* ========================================== */
/* ========================================== */
/* === Polynomial.cpp (Id return 1 exit) ==== */
/* ========================================== */
/* ========================================== */
/* ========================================== */

template <typename eT>
Polynomial<eT>::Polynomial()
{
	head = new term();
}

template <typename eT>
Polynomial<eT>::Polynomial(const Polynomial& x)
{
	head = new term();
	term* p = x.head->next, *q = head, *tmp = NULL;
	while (p)
	{
		tmp = new term(p->pow,p->coe);
		q = (q->next=tmp);
		p = p->next;
	}
} 

template <typename eT>
Polynomial<eT>& Polynomial<eT>::operator=(const Polynomial<eT>& x)
{
	if (this==&x)  return *this;
	clear(0); 
	head = new term;
	term* p = x.head->next, *q = head, *tmp = NULL;
	while (p)
	{
		tmp = new term(p->pow,p->coe);
		q = (q->next=tmp);
		p = p->next;
	}
}

template <typename eT>
Polynomial<eT>::~Polynomial()
{
	term* p = NULL;
	while (head)
	{
		p = head->next;
		delete head;
		head = p;
	}
}

template <typename eT>
void Polynomial<eT>::clear(int)
{
	term* p = NULL;
	while (head)
	{
		p = head->next;
		delete head;
		head = p;
	}
}

template <typename eT>
void Polynomial<eT>::clear()
{
	clear(0);
	head = new term;
}

template <typename eT>
void Polynomial<eT>::addterm(int x, eT y)
{
	if (x<0) throw IllegalPower();
	term* p = head->next, *q = head;
	while (p)
	{
		if (p->pow>=x) break;
		q = p;
		p = p->next;
	}
	if (p && p->pow==x)
	{
		p->coe += y;
		if (!p->coe)
		{
			q->next = p->next;
			delete p;
		}
		return;
	}
	term* tmp = new term(x,y,p);
	q->next = tmp;
}

template <typename eT>
void Polynomial<eT>::add(const Polynomial& x, const Polynomial& y)
{
	Polynomial qwe;
	term* p = x.head->next, *q = y.head->next, *r = qwe.head, *tmp = NULL;
	while (p && q)
	{
		if (p->pow==q->pow)
		{
			if (!(p->coe+q->coe))	{ p = p->next; q = q->next; continue;	}
			tmp = new term(p->pow,p->coe+q->coe);
			r = (r->next = tmp);
			p = p->next;	q = q->next;
			continue;
		}
		if (p->pow<q->pow)
		{
			tmp = new term(p->pow,p->coe);
			r = (r->next = tmp);
			p = p->next;
			continue;
		}
		tmp = new term(q->pow,q->coe);
		r = (r->next = tmp);
		q = q->next;
	}
	while (p)
	{
		tmp = new term(p->pow,p->coe);
		r = (r->next = tmp);
		p = p->next;
		continue;	
	}
	while (q)
	{
		tmp = new term(q->pow,q->coe);
		r = (r->next = tmp);
		q = q->next;
	}
	
	clear(0);
	
	*this = qwe;
}

template <typename eT>
void Polynomial<eT>::print(term* p) const
{
	if (!p) { cout << "0"; return; }
	if (!p->next)
	{
		cout << p->coe << " x^" << p->pow;
		return;
	} 
	print(p->next);
	cout << " + " << p->coe << " x^" << p->pow;
}

template <typename eT>
void Polynomial<eT>::print() const
{
	print(head->next);
	cout << endl;
}

template <typename eT>
eT Polynomial<eT>::value(eT x) const
{
	eT tmp = 0, sum = 1;
	int po = 0;
	term *p = head->next;
	while (p)
	{
		while (po<p->pow) { sum*=x; po++;	}
		tmp += p->coe*sum;
		p = p->next;
	}
	return tmp;
}

#endif

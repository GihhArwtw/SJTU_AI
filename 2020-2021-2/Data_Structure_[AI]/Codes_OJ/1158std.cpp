#include <iostream>
#include <cstring>
using namespace std;
template <class T_KEY, class T_OTHER>

// SET
struct SET
{
	T_KEY key;
	T_OTHER other;
};

// ERROR
class OutOfBoundsException
{
};

// STACKBASE
template <class elemType>
class StackBase
{
  public:
	virtual bool ifempty() const = 0;
	virtual elemType top() const = 0;
	virtual void push(const elemType &x) = 0;
	virtual elemType pop() = 0;
	virtual ~StackBase() {}
};

template <class elemType>
class StacKun : public StackBase<elemType>
{
  private:
	elemType *p, *q;
	int Top;
	int maxsize;
	void doublespace();

  public:
	StacKun(int intsize);
	bool ifempty() const;
	elemType top() const;
	void push(const elemType &x);
	elemType pop();
	~StacKun()
	{
		delete[] p;
		delete[] q;
	}
	void getMin() const;
};

template <class elemType>
StacKun<elemType>::StacKun(int intsize)
{
	maxsize = intsize;
	p = new elemType[intsize];
	q = new elemType[intsize];
	if (!p)
		throw OutOfBoundsException();
	Top = -1;
}
template <class elemType>
bool StacKun<elemType>::ifempty() const
{
	if (Top == -1)
		return true;
	return false;
}
template <class elemType>
elemType StacKun<elemType>::top() const
{
	if (Top == -1)
		throw OutOfBoundsException();
	return p[Top];
}
template <class elemType>
void StacKun<elemType>::push(const elemType &x)
{
	if (Top == maxsize - 1)
		doublespace();
	p[++Top] = x;
}
template <class elemType>
elemType StacKun<elemType>::pop()
{
	if (Top == -1)
		throw OutOfBoundsException();
	int j;
	j = Top;
	Top--;
	return p[j];
}
template <class elemType>
void StacKun<elemType>::doublespace()
{
	elemType *tmp = p;
	elemType *tmq = q;
	p = new elemType[maxsize * 2];
	q = new elemType[maxsize * 2];
	if (!p)
		throw OutOfBoundsException();
	for (int i = 0; i <= Top; i++)
		p[i] = tmp[i];
	for (int i = 0; i <= Top; i++)
		q[i] = tmq[i];
	maxsize = maxsize * 2;
	delete[] tmp;
}
template <class elemType>
void StacKun<elemType>::getMin() const
{
	cout << q[Top];
}

template <class T_KEY, class T_OTHER>
class BinarySearchTree
{
  private:
	struct Node
	{
		SET<T_KEY, T_OTHER> data;
		Node *left;
		Node *right;
		Node(const SET<T_KEY, T_OTHER> &thisData, Node *lt = NULL, Node *rt = NULL) : data(thisData), left(lt), right(rt) {}
	};
	Node *root;
	void insert(const SET<T_KEY, T_OTHER> &x, Node *&t);
	SET<T_KEY, T_OTHER> *find(const T_KEY &x, Node *t);
	void remove(const T_KEY &x, Node *&t);
	void empty(Node *t);
	bool search(const T_KEY &x, Node *t);

  public:
	BinarySearchTree();
	~BinarySearchTree();
	void insert(const SET<T_KEY, T_OTHER> &x);
	void remove(const T_KEY &x);
	bool search(const T_KEY &x);
	SET<T_KEY, T_OTHER> *find(const T_KEY &x);
	bool findAdvanced(const T_KEY &x);
	void insertf(const SET<T_KEY, T_OTHER> &x);
	void removeAdvanced(const T_KEY &x);
	void find_ith(int k);
	bool delete_interval(int a, int b);
};
template <class T_KEY, class T_OTHER>
void BinarySearchTree<T_KEY, T_OTHER>::find_ith(int k)
{
	if (!root)
	{
		cout << "N" << endl;
		return;
	}
	StacKun<Node *> s(10);
	Node *p;
	s.push(root);
	p = root;
	while (p->left)
	{
		s.push(p->left);
		p = p->left;
	}

	while (!s.ifempty())
	{
		p = s.pop();
		k--;
		if (k == 0)
		{
			cout << p->data.key << endl;
			return;
		}
		if (p->right)
		{
			s.push(p->right);
			p = p->right;
			while (p->left)
			{
				s.push(p->left);
				p = p->left;
			}
		}
	}
	cout << "N" << endl;
}
template <class T_KEY, class T_OTHER>
bool BinarySearchTree<T_KEY, T_OTHER>::delete_interval(int a, int b)
{
	if (!root)
		return true;
	StacKun<Node *> s(10);
	Node *p;
	s.push(root);
	p = root;
	int x;
	while (p->left)
	{
		s.push(p->left);
		p = p->left;
	}

	while (!s.ifempty())
	{
		p = s.pop();
		if (a < p->data.key && b > p->data.key)
		{
			x = p->data.key;
			removeAdvanced(x);
			return false;
		}
		if (p->right)
		{
			s.push(p->right);
			p = p->right;
			while (p->left)
			{
				s.push(p->left);
				p = p->left;
			}
		}
	}
	return true;
}
template <class T_KEY, class T_OTHER>
void BinarySearchTree<T_KEY, T_OTHER>::insertf(const SET<T_KEY, T_OTHER> &x)
{
	Node *p;
	if (!root)
	{
		root = new Node(x);
		return;
	}
	p = root;
	while (p)
	{
		if (p->data.key >= x.key)
		{
			if (!p->left)
			{
				p->left = new Node(x);
				return;
			}
			p = p->left;
		}
		else
		{
			if (!p->right)
			{
				p->right = new Node(x);
				return;
			}
			p = p->right;
		}
	}
}

template <class T_KEY, class T_OTHER>
void BinarySearchTree<T_KEY, T_OTHER>::removeAdvanced(const T_KEY &x)
{
	if (!root)
		return;
	Node *p, *parent;
	p = root;
	parent = NULL;
	int flag;
	while (p)
	{
		if (x < p->data.key)
		{
			parent = p;
			flag = 0;
			p = p->left;
			continue;
		}
		if (x > p->data.key)
		{
			parent = p;
			flag = 1;
			p = p->right;
			continue;
		}

		if (!p->left && !p->right)
		{
			delete p;
			if (!parent)
			{
				root = NULL;
				return;
			}
			if (flag == 0)
				parent->left = NULL;
			else
				parent->right = NULL;
			return;
		}
		if (!p->left || !p->right)
		{
			Node *tmp;
			tmp = p;
			if (!parent)
				root = (p->left) ? p->left : p->right;
			else if (flag == 0)
				parent->left = (p->left) ? p->left : p->right;
			else
				parent->right = (p->left) ? p->left : p->right;
			delete tmp;
			return;
		}
		else
		{
			Node *q, *substitute;
			parent = p;
			flag = 0;
			q = p->left;
			while (q->right)
			{
				parent = q;
				flag = 1;
				q = q->right;
			}
			substitute = q;
			SET<T_KEY, T_OTHER> e;
			e = p->data;
			p->data = substitute->data;
			substitute->data = e;
			p = substitute;
		}
	}
}
template <class T_KEY, class T_OTHER>
BinarySearchTree<T_KEY, T_OTHER>::BinarySearchTree()
{
	root = NULL;
}
template <class T_KEY, class T_OTHER>
void BinarySearchTree<T_KEY, T_OTHER>::empty(Node *t)
{
	if (!t)
		return;
	empty(t->left);
	empty(t->right);
	delete t;
}
template <class T_KEY, class T_OTHER>
BinarySearchTree<T_KEY, T_OTHER>::~BinarySearchTree()
{
	empty(root);
}
template <class T_KEY, class T_OTHER>
SET<T_KEY, T_OTHER> *BinarySearchTree<T_KEY, T_OTHER>::find(const T_KEY &x, Node *t)
{
	if (t == NULL || t->data.key == x)
		return (SET<T_KEY, T_OTHER> *)t;
	if (t->data.key > x)
		return find(x, t->left);
	else
		return find(x, t->right);
}
template <class T_KEY, class T_OTHER>
SET<T_KEY, T_OTHER> *BinarySearchTree<T_KEY, T_OTHER>::find(const T_KEY &x)
{
	return find(x, root);
}
template <class T_KEY, class T_OTHER>
bool BinarySearchTree<T_KEY, T_OTHER>::search(const T_KEY &x, Node *t)
{
	if (t == NULL)
		return false;
	if (t->data.key == x)
		return true;
	if (t->data.key > x)
		return search(x, t->left);
	else
		return search(x, t->right);
}
template <class T_KEY, class T_OTHER>
bool BinarySearchTree<T_KEY, T_OTHER>::search(const T_KEY &x)
{
	return search(x, root);
}
template <class T_KEY, class T_OTHER>
void BinarySearchTree<T_KEY, T_OTHER>::insert(const SET<T_KEY, T_OTHER> &x, Node *&t)
{
	if (t == NULL)
	{
		t = new Node(x, NULL, NULL);
		return;
	}
	if (t->data.key == x.key)
		return;
	if (t->data.key > x.key)
		insert(x, t->left);
	else
		insert(x, t->right);
}
template <class T_KEY, class T_OTHER>
void BinarySearchTree<T_KEY, T_OTHER>::insert(const SET<T_KEY, T_OTHER> &x)
{
	insert(x, root);
}
template <class T_KEY, class T_OTHER>
bool BinarySearchTree<T_KEY, T_OTHER>::findAdvanced(const T_KEY &x)
{
	if (!root)
		return NULL;
	Node *p;
	p = root;
	while (p)
	{
		if (p->data.key == x)
			return true;
		if (p->data.key > x)
			p = p->left;
		else
			p = p->right;
	}
	return false;
}
template <class T_KEY, class T_OTHER>
void BinarySearchTree<T_KEY, T_OTHER>::remove(const T_KEY &x, Node *&t)
{
	if (!t)
		return;
	if (t->data.key > x)
		remove(x, t->left);
	else if (t->data.key < x)
		remove(x, t->right);
	else if (t->left && t->right)
	{
		Node *tmp = t->right;
		while (tmp->left)
			tmp = tmp->left;
		t->data = tmp->data;
		remove(tmp->data.key, t->right);
	}
	else
	{
		Node *t1 = t;
		if (t->left)
			t = t->left;
		else
			t = t->right;
		delete t1;
	}
}
template <class T_KEY, class T_OTHER>
void BinarySearchTree<T_KEY, T_OTHER>::remove(const T_KEY &x)
{
	remove(x, root);
}

// Driver function
int main()
{
	freopen("1158.txt","r",stdin);
	freopen("1.txt","w",stdout);
	
	int n, i, j, k, m, n1 = 0, a, b;
	SET<int, int> x;
	BinarySearchTree<int, int> binarySearchTree;
	char ch[20];
	cin >> n;
	cin.getline(ch, 20);
	for (i = 1; i <= n; i++)
	{
		cin >> ch;
		j = strlen(ch);
		if (ch[0] == 'i')
		{
			cin >> k;
			x.key = k;
			binarySearchTree.insertf(x);
		}
		if (ch[0] == 'd' && j == 6)

		{
			cin >> k;
			binarySearchTree.removeAdvanced(k);
		}
		if (j == 15)
		{
			cin >> a >> b;
			bool flag1 = false;
			while (!flag1)
			{
				flag1 = binarySearchTree.delete_interval(a, b);
			}
		}
		if (j == 4)
		{
			cin >> k;
			bool p = binarySearchTree.findAdvanced(k);
			if (p)
				cout << "Y" << endl;
			else
				cout << "N" << endl;
		}
		if (j == 8)
		{
			cin >> m;
			binarySearchTree.find_ith(m);
		}
	}
	
	fclose(stdin);
	fclose(stdout);
	
}

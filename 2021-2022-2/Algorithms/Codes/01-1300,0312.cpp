#pragma GCC optimize(2)
#include <iostream>
using namespace std;

const int N = 4e7 + 1;
int n, k;
int a[N];
void read_input_data() {
    int m;
    cin >> n >> k >> m;
    for (int i = 1; i <= m; i++) {
        cin >> a[i];
    }
    unsigned int z = a[m];
    for (int i = m + 1; i <= n; i++) {
        z ^= z << 13;
        z ^= z >> 17;
        z ^= z << 5;
        a[i] = z & 0x7fffffff;
    }
}

int divide(int l, int r)
{
	int ii=l, jj=r;
	int pivot = a[(l+r)/2];
	a[(l+r)/2] = a[l];	a[l] = pivot;
	while (ii<jj)
	{
		while ((a[jj]>=pivot) && (jj>ii))	jj--;
		if (ii==jj) break;
		a[ii] = a[jj]; ii++;
		while ((a[ii]<=pivot) && (ii<jj))	ii++;
		if (ii==jj) break;
		a[jj] = a[ii]; jj--;
	}
	a[ii] = pivot;
	if (jj==k) return pivot;
	if (jj<k)  return divide(ii+1,r);
	return divide(l,ii-1);
}

int main()
{
	read_input_data();
	cout << divide(1,n);
	return 0;
}

#ifndef GRAPH_H
#define GRAPH_H

class ConflictedEdgeWeight
{
	const char* message;
	public:
		ConflictedEdgeWeight(): message("The weight of an edge is exactly the none-edge flag. Please check inputs.") {}
		const char* what() { return message; }
};

class IllegalSize
{
	const char* message;
	public:
		IllealSize(): message("Not a legal size. Please retry.") {} 
		const char* what() { return message; }
};

class OutOfBound
{
	const char* message;
	public:
		OutOfBound(): message("The visit request is out of bound. Please check inputs.") {}
		const char* what() { return message; }
};

template <typename vT, typename eT>
class Graph
{
	public:
		virtual void insert(const vT&, const vT&, const eT&) = 0;
		virtual void remove(const vT&, const vT&) = 0;
		virtual bool exist (const vT&, const vT&) = 0;
		
		virtual ~Graph() {}
	
	protected:
		int NoV;  // Number of Vertices;
		int NoE;  // Number of Edges
};

template <typename vT, typename eT>
class adjMatrixGraph
{
	private:
		int  NoV;
		int  MaxV;
		int  NoE;
		eT   NoneFlag;    // the symbol showing there is no edges between two vertices
		vT   NotExist;    // the symbol showing there is no vertex
		bool Directed;    // true: Directed; false: Undirected
		
		eT** edge;
		vT*  vert;
		
		int  find(const vT& x)  { int i=0;  while (i<NoV && vert[i]!=x) {i++;}  return i; }
		void empty();
		void copy(const adjMatrixGraph&);
		void doubleSpace();
	
	public:
		adjMatrixGraph(int, const vT v[], const eT&, const vT&, bool=true, int=5);
		adjMatrixGraph(const adjMatrixGraph&);
		adjMatrixGraph& operator=(const adjMatrixGraph&);
		~adjMatrixGraph();
	
		void insert(const vT&, const vT&, const eT&);
		void insert(const vT&);
		void remove(const vT&, const vT&);
		void remove(const vT&);
		bool exist (const vT&, const vT&);
		bool exist (const vT&); 
};

/* ============================================ */
/* ============================================ */
/* ============================================ */
/* ======= Graph.cpp (Id return 1 exit) ======= */
/* ============================================ */
/* ============================================ */
/* ============================================ */

template <typename vT, typename eT>
void adjMatrixGraph::empty()
{
	for (int i=0; i<MaxV; i++)  delete []edge[i];
	delete []edge;
	delete []vert;
	NoV = 0;
	NoE = 0;
}

template <typename vT, typename eT>
void adjMatrixGraph::doubleSpace()
{
	eT** tmp = new eT*[MaxV*2];
	if (!tmp)  throw IllegalSize();
	
	for (int i=0; i<MaxV; i++)
	{
		tmp[i] = new eT[MaxV*2];
		if (!tmp[i])  throw IllgealSize();
		
		for (int j=0; j<MaxV; j++)       tmp[i][j] = edge[i][j];
		for (int j=MaxV; j<MaxV*2; j++)  tmp[i][j] = NoneFlag;
		
		delete []edge[i];
	}
	
	for (int i=MaxV; i<MaxV*2; i++)
	{
		tmp[i] = new eT[MaxV*2];
		if (!tmp[i])   throw IllegalSize();
		
		for (int j=0; j<MaxV*2; j++)     tmp[i][j] = NoneFlag;
	}
	
	delete []edge;
	edge = tmp;
	
	vT* temp = new vT[MaxV*2];
	if (!temp) throw IllegalSize(); 
	
	for (int i=0; i<MaxV; i++)       temp[i] = vert[i];
	for (int i=MaxV; i<MaxV*2; i++)  temp[i] = NotExist;
	
	delete []vert;
	vert = temp;
	
	MaxV *= 2;
}

template <typename vT, typename eT>
adjMatrixGraph::adjMatrixGraph(int vers, const vT v[], const eT& NF, const vT& NE, bool bl, int mv)
{
	NoV      = vers;
	MaxV     = mv;
	NoE      = 0;
	NoneFlag = NF;
	NotExist = NE;
	Directed = bl;
	vert     = new vT[MaxV];
	edge     = new eT*[MaxV];
	
	for (int i=0; i<MaxV; i++)
	{
		edge[i] = new eT[MaxV];
		for (int j=0; j<MaxV; j++)  edge[i][j] = NoneFlag;
	}
	
	for (int i=0; i<NoV; i++)  vert[i] = v[i];	
} 

template <typename vT, typename eT>
void adjMatrixGraph::copy(const adjMatrixGraph& x)
{
	NoV      = x.NoV;
	MaxV     = x.MaxV;
	NoE      = x.NoE;
	NoneFlag = x.NoneFlag;
	Directed = x.Directed;
	vert     = new vT[MaxV];
	edge     = new eT*[MaxV];
	
	for (int i=0; i<MaxV; i++)
	{
		edge[i] = new eT[MaxV];
		for (int j=0; j<MaxV; j++)  edge[i][j] = x.edge[i][j];
	}
	
	for (int i=0; i<NoV; i++)  vert[i] = x.vert[i];
}

template <typename vT, typename eT>
adjMatrixGraph::adjMatrixGarph(const adjMatrixGraph& x) { copy(x); }

template <typename vT, typename eT>
adjMatrixGraph& adjMatrixGraph::operator=(const adjMatrixGraph& x) { empty(); copy(x); }

template <typename vT, typename eT>
void adjMatrixGraph::insert(const vT& x, const vT& y, const eT& w)
{
	if (w==NoneFlag)  throw ConflictedEdgeWeight();
	edge[x][y] = w;
	if (!Directed) edge[y][x] = w;
	
	NoE++;
}

template <typename vT, typename eT>
void adjMatrixGraph::insert(const vT& x)
{
	NoV++;
	if (NoV==MaxV)  doubleSpace();
	
	vert[NoV] = x;	
}

template <typename vT, typename eT>
void adjMatrixGraph::remove(const vT& x, const vT& y)
{
	edge[x][y] = NoneFlag;
	if (!Directed) edge[y][x] = NoneFlag; 
	
	NoE--;
}

template <typename vT, typename eT>
void adjMatrixGraph::remove(const vT& x)
{
	int i = find(x);
	if (i==NoV)  throw OutOfBound();
	
	int count = 0;
	for (int j=0; j<NoV; j++)
	{
		if (edge[i][j]!=NoneFlag)  count++;
		if (edge[j][i]!=NoneFlag)  count++;
	}
	
	for (int j=i; j<NoV-1; j++)  vert[j] = vert[j+1]; 
	for (int j=i; j<NoV-1; j++)
}
	
#endif

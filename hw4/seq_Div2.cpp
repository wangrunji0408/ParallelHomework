#include <stdio.h>
#include <stdlib.h>
#include <cassert>

const int INF = 10000000;
const int V = 10010;
void input(char *inFileName);
void output(char *outFileName);

void seq_FW();
void seq_Div2();

int n, m;	// Number of vertices, edges
static int Dist[V][V];

int main(int argc, char* argv[])
{
	input(argv[1]);
	seq_Div2();

	output(argv[2]);

	return 0;
}

void input(char *inFileName)
{
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &n, &m);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (i == j)	Dist[i][j] = 0;
			else		Dist[i][j] = INF;
		}
	}

	while (--m >= 0) {
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		--a, --b;
		Dist[a][b] = v;
	}
}

void output(char *outFileName)
{
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (Dist[i][j] >= INF)	fprintf(outfile, "INF ");
			else					fprintf(outfile, "%d ", Dist[i][j]);
		}
		fprintf(outfile, "\n");
	}		
}

inline void updateMin (int &x, int a)
{
	if(a < x)	x = a;
}

void seq_FW(int i0, int j0, int k0, int B)
{
	for (int k = k0; k < k0 + B; k++) {
		for (int i = i0; i < i0 + B; i++) {
			for (int j = j0; j < j0 + B; j++) {
				updateMin(Dist[i][j], Dist[i][k] + Dist[k][j]);
			}
		}
	}
}

void seq_Div2 ()
{
	assert(n % 2 == 0);
	int B = n / 2;
	// seq_FW(0, 0, 0, B);
	// seq_FW(B, B, B, B);

	// seq_FW(B, 0, 0, B);
	// seq_FW(B, 0, B, B);

	// seq_FW(0, B, 0, B);
	// seq_FW(0, B, B, B);

	// seq_FW(0, 0, B, B);
	// seq_FW(B, B, 0, B);
	seq_FW(0, 0, 0, B);
	seq_FW(B, 0, 0, B);
	seq_FW(0, B, 0, B);
	seq_FW(B, B, 0, B);
	
	seq_FW(B, B, B, B);
	seq_FW(B, 0, B, B);
	seq_FW(0, B, B, B);
	seq_FW(0, 0, B, B);
	
}

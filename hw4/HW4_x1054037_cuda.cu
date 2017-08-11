#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <vector>
using namespace std;

const int INF = 10000000;
const int V = 10010;
const int MAX_THREAD_DIM2 = 32;
void input(char *inFileName, int B);
void output(char *outFileName);

void block_FW(int B);
int ceil(int a, int b);
void calAsync(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int realn;
int n, m;	// Number of vertices, edges
int* Dist;	// n * n, on host
int* dDist; // n * n, on device

int streamSize;
vector<cudaStream_t> streams;

inline cudaStream_t getIdleStream ()
{
	if(streams.size() == streamSize)
	{
		cudaStream_t stm;
		cudaStreamCreate(&stm);
		streams.push_back(stm);
		streamSize++;
		return stm;
	}
	else
		return streams[streamSize++];
}

inline void syncAllStreams ()
{
	cudaThreadSynchronize();
	streamSize = 0;
}

int main(int argc, char* argv[])
{
	int B = atoi(argv[3]);
	input(argv[1], B);
	
	// if(B > n)
	// {
	// 	B = n;
	// 	cerr << "Warning: B > n. Set B = n.";
	// }
	block_FW(B);

	output(argv[2]);

	return 0;
}

void input(char *inFileName, int B)
{
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &realn, &m);
	n = ceil(realn, B) * B;
	Dist = new int[n * n];

	for (int i = 0, k = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j, ++k) {
			if (i == j)	Dist[k] = 0;
			else		Dist[k] = INF;
		}
	}

	while (--m >= 0) {
		int a, b, v;
		fscanf(infile, "%d %d %d", &a, &b, &v);
		--a, --b;
		Dist[a * n + b] = v;
	}
}

void output(char *outFileName)
{
	FILE *outfile = fopen(outFileName, "w");
	for (int i = 0; i < realn; ++i) {
		for (int j = 0; j < realn; ++j) {
			int d = Dist[i * n + j];
			if (d >= INF)	fprintf(outfile, "INF ");
			else			fprintf(outfile, "%d ", d);
		}
		fprintf(outfile, "\n");
	}		
	delete[] Dist;
}

void print ()
{
	for (int i = 0; i < realn; ++i) {
		for (int j = 0; j < realn; ++j) {
			int d = Dist[i * n + j];
			if (d >= INF)	fprintf(stderr, "INF ");
			else			fprintf(stderr, "%d ", d);
		}
		fprintf(stderr, "\n");
	}	
	fprintf(stderr, "\n");
}

inline int ceil(int a, int b)
{
	return (a + b -1)/b;
}
inline __device__
void updateMin (int &x, int a)
{
	if(a < x)	x = a;
}


__global__
void UpdateIKJ32 (int r, int* dDist, int n)
{
	#define D(i,j) (dDist[(i) * n + (j)])
	int tx = threadIdx.x;
    int ty = threadIdx.y;
	int i = r * 32 + tx;
	int j = r * 32 + ty;
	__shared__ int S[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	S[tx][ty] = D(i, j);
	__syncthreads();
	for(int k=0; k<32; ++k)
	{
		updateMin(S[tx][ty], S[tx][k] + S[k][ty]);
		__syncthreads();
	}
	D(i, j) = S[tx][ty];
	#undef D
}

__global__
void UpdateIK32 (int r, int* dDist, int n)
{
	#define D(i,j) (dDist[(i) * n + (j)])
	int tx = threadIdx.x;
    int ty = threadIdx.y;
	int by = blockIdx.x;
	if(by >= r)	by++;
	int i = r * 32 + tx;
	int j = by * 32 + ty;
	__shared__ int S0[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	__shared__ int S1[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	S0[ty][tx] = D(i, r*32 + ty);
	S1[tx][ty] = D(i, j);
	__syncthreads();
	for(int k=0; k<32; ++k)
	{
		updateMin(S1[tx][ty], S0[k][tx] + S1[k][ty]);
		__syncthreads();
	}
	D(i, j) = S1[tx][ty];
	#undef D
}

__global__
void UpdateKJ32 (int r, int* dDist, int n)
// 0 --update--> 1
{
	#define D(i,j) (dDist[(i) * n + (j)])
	int tx = threadIdx.x;
    int ty = threadIdx.y;
	int bx = blockIdx.x;
	if(bx >= r) bx++;
	int i = bx * 32 + tx;
	int j = r * 32 + ty;
	__shared__ int S0[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	__shared__ int S1[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	S0[ty][tx] = D(i, j);
	S1[tx][ty] = D(r*32 + tx, j);
	__syncthreads();
	for(int k=0; k<32; ++k)
	{
		updateMin(S0[ty][tx], S0[k][tx] + S1[k][ty]);
		__syncthreads();
	}
	D(i, j) = S0[ty][tx];
	#undef D
}

__global__
void Update32 (int r, int* dDist, int n)
{
	#define D(i,j) (dDist[(i) * n + (j)])
	int tx = threadIdx.x;
    int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	if(bx >= r)	bx++;
	if(by >= r)	by++;
	int i = bx * 32 + tx;
	int j = by * 32 + ty;
	__shared__ int S0[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	__shared__ int S1[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	S0[ty][tx] = D(i, r * 32 + ty);
	S1[tx][ty] = D(r * 32 + tx, j);
	__syncthreads();
	int Dij = D(i, j);
	for(int k=0; k<32; ++k)
	{
		updateMin(Dij, S0[k][tx] + S1[k][ty]);
		__syncthreads();
	}
	D(i, j) = Dij;
	#undef D
}

void block_FW(int B)
{
	int *dPivot;
	cudaMalloc(&dDist, sizeof(int) * n * n);
	cudaMalloc(&dPivot, sizeof(int) * B * B);
	cudaMemcpy(dDist, Dist, sizeof(int) * n * n, cudaMemcpyHostToDevice);

	int round = ceil(n, B);
	if(B == 32)
	{
		for (int r = 0; r < round; ++r) 
		{
			/* Phase 1*/
			UpdateIKJ32 <<< 1, dim3(32,32) >>> (r, dDist, n);
			/* Phase 2*/
			UpdateIK32 <<< round-1, dim3(32,32), 0, getIdleStream() >>> (r, dDist, n);
			UpdateKJ32 <<< round-1, dim3(32,32), 0, getIdleStream() >>> (r, dDist, n);
			syncAllStreams();
			/* Phase 3*/
			Update32 <<< dim3(round-1, round-1), dim3(32,32) >>> (r, dDist, n);		
		}
	}
	else
	for (int r = 0; r < round; ++r) {
		/* Phase 1*/
		calAsync(B,	r,	r,	r,	1,	1);
		syncAllStreams();

		/* Phase 2*/
		calAsync(B, r,     r,     0,             r,             1);
		calAsync(B, r,     r,  r +1,  round - r -1,             1);
		calAsync(B, r,     0,     r,             1,             r);
		calAsync(B, r,  r +1,     r,             1,  round - r -1);
		syncAllStreams();

		/* Phase 3*/
		calAsync(B, r,     0,     0,            r,             r);
		calAsync(B, r,     0,  r +1,  round -r -1,             r);
		calAsync(B, r,  r +1,     0,            r,  round - r -1);
		calAsync(B, r,  r +1,  r +1,  round -r -1,  round - r -1);
		syncAllStreams();
	}
	
	cudaMemcpy(Dist, dDist, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
	cudaFree(dDist);
	cudaFree(dPivot);
}

__global__
void Update (int k, int i0, int j0, int i1, int j1, int* dDist, int n)
{
	#define D(i,j) (dDist[(i) * n + (j)])
	int i = blockDim.x * blockIdx.x + threadIdx.x + i0;
    int j = blockDim.y * blockIdx.y + threadIdx.y + j0;
	if(i >= i1 || j >= j1)
		return;
	updateMin(D(i, j), D(i, k) + D(k, j));
}

__global__
void UpdateIndependent (int k0, int k1, int i0, int j0, int i1, int j1, int* dDist, int n)
{
	#define D(i,j) (dDist[(i) * n + (j)])
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int di = blockDim.x * blockIdx.x + tx;
    int dj = blockDim.y * blockIdx.y + ty;
	int i = i0 + di;
	int j = j0 + dj;
	bool valid = i < i1 && j < j1;
	__shared__ int Si[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	__shared__ int Sj[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	const int cacheSize = MAX_THREAD_DIM2;
	int Dij = valid? D(i, j): 0;
	int dkmod = 0;
	for(int k = k0; k < k1; ++k)
	{
		if(dkmod == 0)
		{
			__syncthreads();
			if(i < i1 && k+ty < k1)
			Si[ty][tx] = D(i, k+ty);
			if(j < j1 && k+tx < k1)
			Sj[tx][ty] = D(k+tx, j);
			__syncthreads();		
		}
		if(valid)
		{
			// assert(Si[tx][dkmod] == D(i,k));
			// assert(Sj[dkmod][ty] == D(k,j));			
			
			// int Dik = D(i, k);
			// int Dkj = D(k, j);

			int Dik = Si[dkmod][tx];
			int Dkj = Sj[dkmod][ty];
			updateMin(Dij, Dik + Dkj);
		}
		dkmod = (dkmod + 1) % cacheSize;
	}
	if(valid)
		D(i, j) = Dij;
}

void calAsync(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height)
{
	int block_end_x = block_start_x + block_height;
	int block_end_y = block_start_y + block_width;
	int block_total = block_width * block_height;

	for (int b_i =  block_start_x; b_i < block_end_x; ++b_i) {
		for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
			// To calculate B*B elements in the block (b_i, b_j)
			// For each block, it need to compute B times
			// for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
				// To calculate original index of elements in the block (b_i, b_j)
				// For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
				int i0 = b_i * B;
				int i1 = min((b_i +1) * B, n);
				int j0 = b_j * B; 
				int j1 = min((b_j +1) * B, n);
				int k0 = Round * B;
				int k1 = min((Round +1) * B, n);
				bool iDepends = i0 == k0;
				bool jDepends = j0 == k0;

				int threadDim = MAX_THREAD_DIM2;//std::min(B, MAX_THREAD_DIM2);
				int blockDim = (B + MAX_THREAD_DIM2 - 1) / MAX_THREAD_DIM2;
				dim3 grid(blockDim, blockDim), block(threadDim, threadDim);
				cudaStream_t stm = getIdleStream();
				if(iDepends || jDepends)
				{
					for(int k=k0; k<k1; ++k)
						Update<<<grid, block, 0, stm>>>(k, i0, j0, i1, j1, dDist, n);
				}
				else
					UpdateIndependent<<<grid, block, 0, stm>>>(k0, k1, i0, j0, i1, j1, dDist, n);

				// for (int i = i0; i < i1; ++i) {
				// 	for (int j = j0; j < j1; ++j) {
				// 		if (Dist[i][k] + Dist[k][j] < Dist[i][j])
				// 			Dist[i][j] = Dist[i][k] + Dist[k][j];
				// 	}
				// }
			// }
		}
	}
}



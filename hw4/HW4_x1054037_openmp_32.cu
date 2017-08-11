#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <assert.h>
#include <vector>
using namespace std;

const int INF = 10000000;
const int MAX_THREAD_DIM2 = 32;

int realn;
int n, m;	// Number of vertices, edges
int* Dist;	// n * n, on host
int* dDist[2]; // n * n, on device

int streamSize[2];
vector<cudaStream_t> streams[2];

int getGPUId ()
{
	int gpuId;
	cudaGetDevice(&gpuId);
	return gpuId;
}
cudaStream_t getIdleStream (int gpuId)
{
	cudaSetDevice(gpuId);
	if(streams[gpuId].size() == streamSize[gpuId])
	{
		cudaStream_t stm;
		cudaStreamCreate(&stm);
		streams[gpuId].push_back(stm);
		streamSize[gpuId]++;
		return stm;
	}
	else
		return streams[gpuId][streamSize[gpuId]++];
}
cudaStream_t getIdleStream ()
{
	return getIdleStream(getGPUId());
}
void syncAllStreams (int gpuId)
{
	cudaSetDevice(gpuId);
	cudaThreadSynchronize();
	streamSize[gpuId] = 0;
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

void input(char *inFileName)
{
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &realn, &m);
	n = ceil(realn, 64) * 64;
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

const bool H2D = true;
const bool D2H = false;
cudaStream_t blockCopyAsync (int gpuId, bool h2d, int i0, int j0, int Bi, int Bj)
{
	// cudaSetDevice(gpuId);
	cudaStream_t stream = getIdleStream(gpuId);
	int *dst = dDist[gpuId];
	int *src = Dist;
	if(!h2d)	swap(dst, src);
	cudaMemcpyKind kind = h2d? cudaMemcpyHostToDevice: cudaMemcpyDeviceToHost;
	for(int i = i0; i < i0 + Bi; ++i)
	{
		int offset = i * n + j0;
		int size = Bj * sizeof(int);
		cudaMemcpyAsync(dst + offset, src + offset, size, kind, stream);
	}
	return stream;
}
cudaStream_t blockCopyAsync (int gpuId, bool h2d, int i0, int j0, int B)
{
	return blockCopyAsync(gpuId, h2d, i0, j0, B, B);
}
cudaStream_t blockCopyHalfAsync (int gpuId, bool h2d, int i0, int j0, int B, int half)
{
	int io = B/2;
	if(half == 0)
		return blockCopyAsync(gpuId, h2d, i0, j0, io, B);
	else
		return blockCopyAsync(gpuId, h2d, i0+io, j0, io, B);
}

__global__
void UpdateIKJ32 (int k0, int r, int* dDist, int n)
{
	#define D(i,j) (dDist[(i) * n + (j)])
	int tx = threadIdx.x;
    int ty = threadIdx.y;
	int i = k0 + r * 32 + tx;
	int j = k0 + r * 32 + ty;
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
void UpdateIK32 (int k0, int j0, int r, int* dDist, int n)
{
	#define D(i,j) (dDist[(i) * n + (j)])
	int tx = threadIdx.x;
    int ty = threadIdx.y;
	int by = blockIdx.x;
	int i = k0 + r * 32 + tx;
	int j = j0 + by * 32 + ty;
	__shared__ int S0[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	__shared__ int S1[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	S0[ty][tx] = D(i, k0 + r*32 + ty);
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
void UpdateKJ32 (int k0, int i0, int r, int* dDist, int n)
{
	#define D(i,j) (dDist[(i) * n + (j)])
	int tx = threadIdx.x;
    int ty = threadIdx.y;
	int bx = blockIdx.x;
	int i = i0 + bx * 32 + tx;
	int j = k0 + r * 32 + ty;
	__shared__ int S0[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	__shared__ int S1[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	S0[ty][tx] = D(i, j);
	S1[tx][ty] = D(k0 + r*32 + tx, j);
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
void Update32 (int k0, int i0, int j0, int r, int* dDist, int n)
{
	#define D(i,j) (dDist[(i) * n + (j)])
	int tx = threadIdx.x;
    int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int i = i0 + bx * 32 + tx;
	int j = j0 + by * 32 + ty;
	__shared__ int S0[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	__shared__ int S1[MAX_THREAD_DIM2][MAX_THREAD_DIM2];
	S0[ty][tx] = D(i, k0 + r * 32 + ty);
	S1[tx][ty] = D(k0 + r * 32 + tx, j);
	int Dij = D(i, j);
	__syncthreads();
	for(int k=0; k<32; ++k)
		updateMin(Dij, S0[k][tx] + S1[k][ty]);
	D(i, j) = Dij;
	#undef D
}

void UpdateBlock (int gpuId, int k0, int i0, int j0, int B) // B % 32 == 0
{
	// cudaSetDevice(gpuId);
	assert(B % 32 == 0);
	int round = B / 32;
	for (int r = 0; r < round; ++r) 
	{
		/* Phase 1*/
		if(i0 == k0 && k0 == j0)
		UpdateIKJ32 <<< 1, dim3(32,32) >>> (k0, r, dDist[gpuId], n);
		/* Phase 2*/
		if(i0 == k0)
		UpdateIK32 <<< round, dim3(32,32), 0, getIdleStream() >>> (k0, j0, r, dDist[gpuId], n);
		if(k0 == j0)
		UpdateKJ32 <<< round, dim3(32,32), 0, getIdleStream() >>> (k0, i0, r, dDist[gpuId], n);
		syncAllStreams(gpuId);
		/* Phase 3*/
		Update32 <<< dim3(round, round), dim3(32,32) >>> (k0, i0, j0, r, dDist[gpuId], n);		
	}
}

void UpdateBlockHalf (int gpuId, int k0, int i0, int j0, int B, int half) // B % 32 == 0
{
	int round = B / 32;
	if(half == 0)
	for (int r = 0; r < round; ++r) 
		Update32 <<< dim3((round+1)/2, round), dim3(32,32) >>> (k0, i0, j0, r, dDist[gpuId], n);
	else
	for (int r = 0; r < round; ++r) 
		Update32 <<< dim3((round+1)/2, round), dim3(32,32) >>> (k0, i0+round/2*32, j0, r, dDist[gpuId], n);
}

void block_FW()
{
	int B = n / 2;
	#pragma omp parallel num_threads(2)
	{
		int gpuId = omp_get_thread_num(); 
		cudaSetDevice(gpuId);
		cudaMalloc(&dDist[gpuId], sizeof(int) * n * n);
		blockCopyAsync(gpuId, H2D, 0, 0, n);
		syncAllStreams(gpuId);
		// 1
		UpdateBlock(gpuId,  0, 0, 0, B);
		// 2 3
		if(gpuId == 0)	UpdateBlock(0,  0, B, 0, B);
		else			UpdateBlock(1,  0, 0, B, B);
		syncAllStreams(gpuId);	
		if(gpuId == 0)	blockCopyAsync(0, D2H, B, 0, B);
		else			blockCopyAsync(1, D2H, 0, B, B);
		syncAllStreams(gpuId);
		#pragma omp barrier
		if(gpuId == 0)	blockCopyAsync(0, H2D, 0, B, B);
		else			blockCopyAsync(1, H2D, B, 0, B);
		syncAllStreams(gpuId);
		// 4
		UpdateBlock(gpuId,  0, B, B, B);
		// 5
		UpdateBlock(gpuId,  B, B, B, B);
		// 6 7
		if(gpuId == 0)	UpdateBlock(0,  B, B, 0, B);
		else			UpdateBlock(1,  B, 0, B, B);
		syncAllStreams(gpuId);		
		if(gpuId == 0)	blockCopyAsync(0, D2H, B, 0, B);
		else			blockCopyAsync(1, D2H, 0, B, B);
		syncAllStreams(gpuId);
		#pragma omp barrier
		if(gpuId == 0)	blockCopyAsync(0, H2D, 0, B, B);
		else			blockCopyAsync(1, H2D, B, 0, B);
		syncAllStreams(gpuId);
		// 8
		UpdateBlock(gpuId,  B, 0, 0, B);
		syncAllStreams(gpuId);		

		if(gpuId == 0)
		{
			blockCopyAsync (0, D2H, 0, 0, n);
			syncAllStreams(0);
		}
		cudaFree(dDist[gpuId]);
	}
}

int main(int argc, char* argv[])
{
	int B = atoi(argv[3]);
	input(argv[1]);
	block_FW();
	output(argv[2]);

	return 0;
}
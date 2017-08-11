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

void block_FW_2GPU(int B);
int ceil(int a, int b);
void calAsync(int gpuId, int B, int Round, int bi0, int bi1, int bj0, int bj1, int half = 0);

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
void syncAllStreams ()
{
	cudaSetDevice(0);
	cudaThreadSynchronize();
	cudaSetDevice(1);
	cudaThreadSynchronize();
	streamSize[0] = 0;
	streamSize[1] = 0;
}
cudaEvent_t appendEvent (cudaStream_t stm)
{
	cudaEvent_t end;
	cudaEventCreate(&end);
	cudaEventRecord(end, stm);
	return end;
}
cudaStream_t newBranchStream (int gpuId, cudaStream_t stm)
{
	// cudaSetDevice(gpuId);
	cudaStream_t stm1 = getIdleStream(gpuId);
	cudaStreamWaitEvent(stm1, appendEvent(stm), 0);
	return stm1;
}

const bool H2D = true;
const bool D2H = false;
cudaStream_t blockCopyAsync (int gpuId, bool h2d, int B, int bi0, int bi1, int bj0, int bj1, int half = 0)
{
	// cudaSetDevice(gpuId);
	cudaStream_t stream = getIdleStream(gpuId);
	int *dst = dDist[gpuId];
	int *src = Dist;
	if(!h2d)	swap(dst, src);
	cudaMemcpyKind kind = h2d? cudaMemcpyHostToDevice: cudaMemcpyDeviceToHost;
	for(int i = bi0 * B; i < bi1 * B; ++i)
	{
		int bi = i / B;
		int offset = i * n + bj0 * B;
		int size = (bj1 - bj0) * B * sizeof(int);
		if(half == 1)
			offset = i * n + max(bi, bj0) * B,
			size = (bj1 - max(bi, bj0)) * B * sizeof(int);
		else if(half == 2)
			size = (min(bi, bj1) - bj0) * B * sizeof(int);
		cudaMemcpyAsync(dst + offset, src + offset, size, kind, stream);
	}
	return stream;
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
	block_FW_2GPU(B);

	output(argv[2]);

	return 0;
}

void input(char *inFileName, int B)
{
	FILE *infile = fopen(inFileName, "r");
	fscanf(infile, "%d %d", &realn, &m);
	n = ceil(realn, B) * B;
	cudaMallocHost(&Dist, n * n * sizeof(int));

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
	cudaFreeHost(Dist);
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

int ceil(int a, int b)
{
	return (a + b -1)/b;
}

void block_FW_2GPU(int B)
{
	for(int t=0; t<2; ++t)
	{
		cudaSetDevice(t);
		cudaMalloc(&dDist[t], sizeof(int) * n * n);
		cudaMemcpy(dDist[t], Dist, sizeof(int) * n * n, cudaMemcpyHostToDevice);
	}

	int round = ceil(n, B);
	for (int r = 0; r < round; ++r) {
		#define PIVOT	r, r+1, r, r+1
		#define LPR		r, r+1, 0, round
		#define LP		r, r+1, 0, r+1		
		#define UPD		0, round, r, r+1
		#define LEFT	r, r+1, 0, r
		#define RIGHT	r, r+1, r+1, round
		#define UP		0, r, r, r+1
		#define DOWN	r+1, round, r, r+1
		#define LU		0, r, 0, r
		#define RD		r+1, round, r+1, round
		#define LD		r+1, round, 0, r
		#define RU		0, r, r+1, round

		// fprintf(stderr, "Round: %d\n", r);
		
		/* Phase 1*/
		calAsync(0, B,	r,	PIVOT);
		calAsync(1, B,	r,	PIVOT);
		syncAllStreams();

		/* Phase 2*/
		calAsync(0, B, r, LEFT);
		calAsync(0, B, r, DOWN);
		calAsync(1, B, r, UP);
		calAsync(1, B, r, RIGHT);
		syncAllStreams();

		// + 0.2s		
		calAsync(0, B, r, LD);
		calAsync(1, B, r, RU);
		blockCopyAsync(0, D2H, B, LP);
		blockCopyAsync(0, D2H, B, DOWN);
		blockCopyAsync(1, D2H, B, UP);
		blockCopyAsync(1, D2H, B, RIGHT);
		syncAllStreams();

		// + 0.1s
		blockCopyAsync(1, H2D, B, LEFT);
		blockCopyAsync(1, H2D, B, DOWN);
		blockCopyAsync(0, H2D, B, UP);
		blockCopyAsync(0, H2D, B, RIGHT);
		syncAllStreams();

		/* Phase 3*/
		// + 0.18s
		calAsync(0, B, r, LU);
		calAsync(1, B, r, RD);
		blockCopyAsync(0, D2H, B, LD);
		blockCopyAsync(1, D2H, B, RU);
		syncAllStreams();

		// + 0.6s
		blockCopyAsync(0, D2H, B, LU);
		blockCopyAsync(1, D2H, B, RD);
		syncAllStreams();

		if(r == round - 1)
			break;
		// + 0.17s
		blockCopyAsync(1, H2D, B, LU);
		blockCopyAsync(0, H2D, B, RD);
		blockCopyAsync(1, H2D, B, LD);
		blockCopyAsync(0, H2D, B, RU);
	}
	syncAllStreams();

	// cudaMemcpy(Dist, dDist, sizeof(int) * n * n, cudaMemcpyDeviceToHost);
	for(int t=0; t<2; ++t)
	{
		cudaSetDevice(t);
		cudaFree(&dDist[t]);
	}
}


__global__
void Update (int k, int i0, int j0, int i1, int j1, int* dDist, int n)
{
	#define D(i,j) (dDist[(i) * n + (j)])
	int i = blockDim.x * blockIdx.x + threadIdx.x + i0;
    int j = blockDim.y * blockIdx.y + threadIdx.y + j0;
	if(i >= i1 || j >= j1)
		return;
	int Dik = D(i, k);
	int Dkj = D(k, j);
	int D1 = Dik + Dkj;
	if (D1 < D(i, j))
		D(i, j) = D1;
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
			int D1 = Dik + Dkj;
			if (D1 < Dij)
				Dij = D1;
		}
		dkmod = (dkmod + 1) % cacheSize;
	}
	if(valid)
		D(i, j) = Dij;
}

void calAsync(int gpuId, int B, int Round, int bi0, int bi1, int bj0, int bj1, int half)
{
	cudaSetDevice(gpuId);

	for(int bi = bi0; bi < bi1; ++bi)
		for(int bj = bj0; bj < bj1; ++bj)
		{
			if(half == 1 && bi > bj)
				continue;
			if(half == 2 && bi <= bj)
				continue;
			int i0 = bi * B;
			int i1 = min((bi +1) * B, n);
			int j0 = bj * B; 
			int j1 = min((bj +1) * B, n);
			int k0 = Round * B;
			int k1 = min((Round +1) * B, n);
			bool iDepends = i0 == k0;
			bool jDepends = j0 == k0;

			int threadDim = MAX_THREAD_DIM2;//std::min(B, MAX_THREAD_DIM2);
			int blockDim = (B + MAX_THREAD_DIM2 - 1) / MAX_THREAD_DIM2;
			dim3 grid(blockDim, blockDim), block(threadDim, threadDim);
			cudaStream_t stm = getIdleStream(gpuId);
			if(iDepends || jDepends)
			{
				for(int k=k0; k<k1; ++k)
					Update<<<grid, block, 0, stm>>>(k, i0, j0, i1, j1, dDist[gpuId], n);
			}
			else
				UpdateIndependent<<<grid, block, 0, stm>>>(k0, k1, i0, j0, i1, j1, dDist[gpuId], n);
		}
}



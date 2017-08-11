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
void calAsync(int gpuId, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

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
	cudaThreadSynchronize();
	streamSize[0] = 0;
	streamSize[1] = 0;
}




void blockCopyAsync (int gpuId, int* dst, const int* src, cudaMemcpyKind kind, cudaStream_t stream, int B, int bi0, int bi1, int bj0, int bj1)
{
	cudaSetDevice(gpuId);
	for(int i = bi0 * B; i < bi1 * B; ++i)
	{
		int offset = i * n + bj0 * B;
		int size = (bj1 - bj0) * B * sizeof(int);
		cudaMemcpyAsync(dst + offset, src + offset, size, kind, stream);
	}
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
		/* Phase 1*/
		fprintf(stderr, "Round: %d\n", r);
		calAsync(0, B,	r,	r,	r,	1,	1);
		syncAllStreams();
		blockCopyAsync(0, Dist, dDist[0], cudaMemcpyDeviceToHost, getIdleStream(0), B, r, r+1, r, r+1); // P 0->H
		syncAllStreams();
		blockCopyAsync(1, dDist[1], Dist, cudaMemcpyHostToDevice, getIdleStream(1), B, r, r+1, r, r+1); // P H->1
		syncAllStreams();

		/* Phase 2*/
		calAsync(0, B, r,     r,     0,             r,             1); // L 0
		calAsync(0, B, r,     r,  r +1,  round - r -1,             1); // R 0
		calAsync(1, B, r,     0,     r,             1,             r); // U 1
		calAsync(1, B, r,  r +1,     r,             1,  round - r -1); // D 1
		syncAllStreams();
		blockCopyAsync(0, Dist, dDist[0], cudaMemcpyDeviceToHost, getIdleStream(0), B, r, r+1, 0, round); // LR 0->H
		blockCopyAsync(1, Dist, dDist[1], cudaMemcpyDeviceToHost, getIdleStream(1), B, 0, round, r, r+1); // UD 1->H
		syncAllStreams();
		blockCopyAsync(1, dDist[1], Dist, cudaMemcpyHostToDevice, getIdleStream(1), B, r, r+1, 0, round); // LR H->1
		blockCopyAsync(0, dDist[0], Dist, cudaMemcpyHostToDevice, getIdleStream(0), B, 0, round, r, r+1); // UD H->0
		syncAllStreams();

		/* Phase 3*/
		calAsync(0, B, r,     0,     0,            r,             r); // <^
		calAsync(1, B, r,     0,  r +1,  round -r -1,             r); // ^>
		calAsync(1, B, r,  r +1,     0,            r,  round - r -1); // <v
		calAsync(0, B, r,  r +1,  r +1,  round -r -1,  round - r -1); // v>
		syncAllStreams();
		blockCopyAsync(0, Dist, dDist[0], cudaMemcpyDeviceToHost, getIdleStream(0), B, 0, r, 0, r); // <^ 0->H
		blockCopyAsync(0, Dist, dDist[0], cudaMemcpyDeviceToHost, getIdleStream(0), B, r+1, round, r+1, round); // v> 0->H
		blockCopyAsync(1, Dist, dDist[1], cudaMemcpyDeviceToHost, getIdleStream(1), B, r+1, round, 0, r); // <v 1->H
		blockCopyAsync(1, Dist, dDist[1], cudaMemcpyDeviceToHost, getIdleStream(1), B, 0, r, r+1, round); // ^> 1->H
		syncAllStreams();
		if(r == round - 1)
			break;
		blockCopyAsync(1, dDist[1], Dist, cudaMemcpyHostToDevice, getIdleStream(1), B, 0, r, 0, r); // <^ H->1
		blockCopyAsync(1, dDist[1], Dist, cudaMemcpyHostToDevice, getIdleStream(1), B, r+1, round, r+1, round); // v> H->1
		blockCopyAsync(0, dDist[0], Dist, cudaMemcpyHostToDevice, getIdleStream(0), B, r+1, round, 0, r); // <v H->0
		blockCopyAsync(0, dDist[0], Dist, cudaMemcpyHostToDevice, getIdleStream(0), B, 0, r, r+1, round); // ^> H->0
		syncAllStreams();
	}

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

void calAsync(int gpuId, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height)
{
	cudaSetDevice(gpuId);

	int block_end_x = block_start_x + block_height;
	int block_end_y = block_start_y + block_width;

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
				cudaStream_t stm = getIdleStream(gpuId);
				if(iDepends || jDepends)
				{
					for(int k=k0; k<k1; ++k)
						Update<<<grid, block, 0, stm>>>(k, i0, j0, i1, j1, dDist[gpuId], n);
				}
				else
					UpdateIndependent<<<grid, block, 0, stm>>>(k0, k1, i0, j0, i1, j1, dDist[gpuId], n);

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



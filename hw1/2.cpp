#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <limits>
#include <cstring>
#include <mpi.h>
#include <ctime>
using namespace std;

// const float inf = numeric_limits<float>::max();
const float inf = 1e38;

int id, size;
int n, m;
int offset, read_size;
const char *input, *output;
float *data0, *data1, *data;
int tio, tcomp, tcomm, treduce;

void print ()
{
	cout << id << ": ";
	for(int i=0; i<m*2; ++i)
		cout << data[i] << " ";
	cout << endl;
}

void sync ()
{
	MPI_Barrier(MPI_COMM_WORLD);
}

void sync_print()
{
	for(int i=0; i<size; ++i)
	{
		if(i == id)
			print();
		sync();
	}
}

void init (int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	n = atoi(argv[1]);
	input = argv[2];
	output = argv[3];
	m = ((n+1) / 2 - 1) / size + 1;
	offset = m * 2 * id;
	read_size = max(0, min(n - offset, m * 2));
}

void small ()
{
	if(id != 0)
		return;
	// read
	FILE* fin = fopen(input, "rb");
	float *data = new float[n];
	fread(data, sizeof(float), n, fin);
	fclose(fin);
	// sort
	sort(data, data + n);
	// write
	FILE* fout = fopen(output, "wb");
	fwrite(data, sizeof(float), n, fout);
	fclose(fout);
}

void read ()
{
	clock_t t0 = clock();

	float* data_all = NULL;
	int size_all = m * 2 * size;
	if(id == 0)
	{
		data_all = new float[size_all];
		FILE* file = fopen(input, "rb");
		fread(data_all, sizeof(float), n, file);
		fclose(file);
		for(int i=n; i<size_all; ++i)
			data_all[i] = inf;
	}
	MPI_Scatter(data_all, m*2, MPI_FLOAT, data, m*2, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if(id == 0)
		delete[] data_all;

	tio += clock() - t0;
}

void write ()
{
	clock_t t0 = clock();
	int size_all = m * 2 * size;
	float* data_all = id == 0? new float[size_all]: NULL;
	MPI_Gather(data, m*2, MPI_FLOAT, data_all, m*2, MPI_FLOAT, 0, MPI_COMM_WORLD);
	if(id == 0)
	{
		FILE* file = fopen(output, "wb");
		fwrite(data_all, sizeof(float), n, file);
		fclose(file);
		delete[] data_all;
	}
	tio += clock() - t0;
}

void move_left() 
{
	int dest = id == size-1? 0: id + 1;
	int source = id == 0? size - 1: id - 1;
	memcpy(data1, data, m * sizeof(float));
	clock_t t0 = clock();
	MPI_Request req;
	MPI_Status status;
	MPI_Isend(data + m, m, MPI_FLOAT, dest, 0, MPI_COMM_WORLD, &req);
	MPI_Recv(data0, m, MPI_FLOAT, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	MPI_Wait(&req, &status);
	tcomm += clock() - t0;
}

void move_right() 
{
	int source = id == size-1? 0: id + 1;
	int dest = id == 0? size - 1: id - 1;
	memcpy(data0, data + m, m * sizeof(float));
	clock_t t0 = clock();
	MPI_Request req;
	MPI_Status status;
	MPI_Isend(data, m, MPI_FLOAT, dest, 0, MPI_COMM_WORLD, &req);
	MPI_Recv(data1, m, MPI_FLOAT, source, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	MPI_Wait(&req, &status);
	tcomm += clock() - t0;
}

bool isNotDecrease (float data[], int n)
{
	for(int i=1; i<n; ++i)
		if(data[i] < data[i-1])
			return false;
	return true;
}

// 返回是否已经排好，假设data0 <= data1
bool merge ()
{
	clock_t t0 = clock();
	merge(data0, data0 + m, data1, data1 + m, data);
	tcomp += clock() - t0;
	return data0[m-1] <= data1[0];
}

bool copy ()
{
	clock_t t0 = clock();
	memcpy(data, data0, m * sizeof(float));
	memcpy(data + m, data1, m * sizeof(float));
	tcomp += clock() - t0;
	return true;
}

bool reduceFinish (bool finish)
{
	bool res;
	clock_t t0 = clock();
	MPI_Allreduce(&finish, &res, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
	treduce += clock() - t0;
	return res;
}

int main (int argc, char* argv[])
{
	init(argc, argv);

	if(n <= size*2)
	{
		small();
		MPI_Finalize();
		return 0;
	}

	data0 = new float[m];
	data1 = new float[m];
	data = new float[m*2];
	
	read();
	sort(data, data + m*2);
	// sync_print();
	for(int cnt = 0; true; )
	{
		move_right();
		bool finish = id == size - 1? copy(): merge();
		bool end = reduceFinish(finish);
		cnt = end? cnt + 1: 0;
		if(cnt == 2)	break;

		move_left();
		finish = merge();
		end = reduceFinish(finish);
		cnt = end? cnt + 1: 0;
		if(cnt == 2)	break;
		// sync_print();
	}
	write();
#define DEBUG
#ifdef DEBUG
	cerr << id << " time: "
		 << "\nio:\t" << tio
		 << "\ncomp:\t" << tcomp
		 << "\ncomm:\t" << tcomm
		 << "\nreduce:\t" << treduce << endl << endl;
#endif

	delete[] data;
	delete[] data0;
	delete[] data1;
	MPI_Finalize();
}

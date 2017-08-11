#include <iostream>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <ctime>

using std::vector;
using std::cout;
using std::cerr;
using std::endl;

typedef std::complex<float> cf32;
typedef std::complex<double> cf64;
typedef unsigned int u32;
typedef unsigned char u8;
const int MAX_ITERATION = 100000;
const int MAX_THREAD_NUM = 12;

int id, size;

int calcIter (cf64 c)
{
	cf64 x = c;
	for(int i=1; i<MAX_ITERATION; ++i)
	{
		double d = x.real()*x.real() + x.imag()*x.imag(); 
		if(d >= 4)
			return i;
		// x = x * x + c; // Too SLOW
		double real = x.real()*x.real() - x.imag()*x.imag() + c.real();
		double imag = 2*x.real()*x.imag() + c.imag();
		x = cf64(real, imag);
	}
	return MAX_ITERATION;
}

struct Task
{
	cf64 anchor;
	double dx, dy;
	u32 width, height, step;
	u32 i, j;
	u32 rest;

	bool has_next () const
	{
		return rest > 0;
	}

	cf64 get_next ()
	{
		cf64 res = anchor + cf64(i * dx, j * dy);
		move(step);
		rest--;
		if(i >= height)
			rest = 0;
		return res;
	}

	void move (int k)
	{
		j += k;
		while(j >= height)
			j -= height, i ++;
	}

	vector<Task> splitN (int n) const
	{
		auto list = vector<Task>(n, *this);
		for(int i=0; i<n; ++i)
		{
			list[i].move(step * i);
			list[i].step *= n;
			list[i].rest = (list[i].rest + n-1) / n;
		}
		return list;
	}
};

void init (int argc, char* argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
}

void work (Task task, int *data)
{
	int i = 0;
	while(task.has_next())
	{
		cf64 c = task.get_next();
		*data = calcIter(c);
		data++;
	}
}

void write (const int* data, int width, int height, const char* output)
{
	FILE *fp = fopen(output, "wb");
	fwrite(&width, sizeof(int), 1, fp);
	fwrite(&height, sizeof(int), 1, fp);
	fwrite(data, sizeof(int), width * height , fp);
	fclose(fp);
}

void work_on_threads (Task task, int* data_proc, int thread_num)
{
	assert(thread_num <= MAX_THREAD_NUM);
	omp_set_num_threads(thread_num);
	auto tasks_thread = task.splitN(thread_num);
#pragma omp parallel for
	for(int i=0; i<thread_num; ++i)
	{
		auto task = tasks_thread[i];
		int* data = new int[task.rest];
		work(task, data);
		for(int j=0, k=i; j<task.rest; ++j, k += thread_num)
			data_proc[k] = data[j];
		delete[] data;
		cerr << clock() << " us: Process " << id << " Thread " << i << " Finished." << endl;
	}
}

int main (int argc, char* argv[])
{
	init(argc, argv);
	int thread_num = atoi(argv[1]);
	double l = atof(argv[2]);
	double r = atof(argv[3]);
	double d = atof(argv[4]);
	double u = atof(argv[5]);
	int x_total = atoi(argv[6]);
	int y_total = atoi(argv[7]);
	const char* output = argv[8];
	
	const int n = x_total * y_total;
	int n_proc = n / size + 1;
	auto task_all = Task {cf64(l, d), (r-l) / x_total, (u-d) / y_total, x_total, y_total, 1, 0, 0, n};
	auto tasks_proc = task_all.splitN(size);
	
	int* data_proc = new int[n_proc];
	int* buffer = id == 0? new int[n_proc * size]: nullptr;
	work_on_threads(tasks_proc[id], data_proc, thread_num);
	MPI_Gather(data_proc, n_proc, MPI_INT, buffer, n_proc, MPI_INT, 0, MPI_COMM_WORLD);
	delete[] data_proc;

	if(id == 0)
	{
		int* data_all = new int[n];
		#pragma omp parallel for
		for(int i=0; i<size; ++i)
		{
			const int* datai = buffer + n_proc * i;
			for(int j=0, k=i; j<tasks_proc[i].rest; ++j, k += size)
				data_all[k] = datai[j];
		}
		write(data_all, x_total, y_total, output);
		delete[] buffer;
		delete[] data_all;
		cerr << clock() << " us: Process 0 Write end." << endl;
	}
	MPI_Finalize();
	return 0;
}
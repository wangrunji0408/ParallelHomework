#include <iostream>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <omp.h>
#include <mpi.h>
#include <mutex>
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

inline bool fake (cf64 c)
{
	double x = c.real(), y = c.imag();
	double y2 = y * y;
	double q = (x - 1.0/4) * (x - 1.0/4) + y2;
	return q * (q + x - 1.0/4) < y2 / 4;
}

int calcIter (cf64 c)
{
	if(fake(c))
		return MAX_ITERATION;
	cf64 x = c;
	for(int i=1; i<MAX_ITERATION; ++i)
	{
		double d = x.real()*x.real() + x.imag()*x.imag(); 
		if(d > 4)
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
	u32 idInSuper, stepInSuper;
	int rest;

	bool goNext ()
	{
		move(step);
		idInSuper += stepInSuper;
		rest--;
		return rest > 0 && i < width;
	}

	cf64 getPoint () const
	{
		return anchor + cf64(i * dx, j * dy);
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
			Task& task = list[i];
			task.move(step * i);
			task.step *= n;
			task.rest = (task.rest + n-1-i) / n;
			task.idInSuper = i;
			task.stepInSuper = n;
		}
		return list;
	}

	vector<Task> splitN_Seq (int n) const
	{
		auto list = vector<Task>(n, *this);
		int offset = 0;
		for(int i=0; i<n; ++i)
		{
			Task& task = list[i];
			task.move(step * offset);
			task.rest = (task.rest + n-1-i) / n;
			task.idInSuper = offset;
			task.stepInSuper = 1;
			offset += task.rest;
		}
		assert(offset == rest);
		return list;
	}

	Task take (int k)
	{
		auto task = *this;
		if(rest > k)
			task.rest = k;

		rest -= k;
		move(step * k);
		idInSuper += stepInSuper * k;
		return task;
	}
};

class TaskScheduler
{
	static const int MASTER_RANK = 0;
public:
	static void handleResult (const char* buffer, int* data)
	{
		Task& task = *(Task*)buffer;
		auto bufint = (const int*)(buffer + sizeof(Task));
		do {
			data[task.idInSuper] = *(bufint++);
		}
		while(task.goNext());
	}
	static Task getNextTask ()
	{
		// cerr << id << " try to get next." << endl;
		Task task;
		MPI_Status status;
		MPI_Recv(&task, sizeof(Task), MPI_BYTE, MASTER_RANK, 0, MPI_COMM_WORLD, &status);
		// cerr << id << " get task:" << task.i << ' ' << task.j << endl;
		return task;
	}
	static void sendTaskResult (const char* buffer, int bufferSize)
	{
		// cerr << id << " try to send result." << endl;
		MPI_Send(buffer, bufferSize, MPI_BYTE, MASTER_RANK, 0, MPI_COMM_WORLD);
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
	do {
		cf64 c = task.getPoint();
		data[task.idInSuper] = calcIter(c);
	}
	while(task.goNext());
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
	auto tasks_thread = task.splitN(thread_num);
#pragma omp parallel for num_threads(thread_num) schedule(static, 1)
	for(int i=0; i<thread_num; ++i)
	{
		work(tasks_thread[i], data_proc);
		// cerr << clock() << " us: Process " << id << " Thread " << i << " Finished." << endl;
	}
}

void dynamicMPI (Task task_all, int thread_num, const char* output)
{
	// omp_set_nested (true);
	// assert(omp_get_nested() == true);
	const int BATCH_SIZE = 1000;
	const int SMALL_SIZE = 100;
	vector<char> buffer;
	if (id == 0)
	{
		int* data = new int[task_all.rest];
		auto emptyTask = Task{0};

		vector<char> buffer;

		int i = 0, rest = 0;
		// cerr << "* begin send all." << endl;

		for(int j=1; j<size; ++j)
		{
			auto task = task_all.take(BATCH_SIZE);
			MPI_Send(&task, sizeof(Task), MPI_BYTE, j, 0, MPI_COMM_WORLD);
			rest += 1;
		}
			
		// cerr << "* begin listening." << endl;
		int count;
		MPI_Request request;
		MPI_Status status;
		int flag;
		bool first = true;
		for(; rest; )
		{
			auto task = task_all.take(SMALL_SIZE);
			if(task.rest >= 0)
			{
				int dataSize = task.rest * sizeof(int) + sizeof(task);
				if(dataSize > buffer.capacity())
					buffer.reserve(dataSize);
				*(Task*)buffer.data() = task;
				work_on_threads(task, (int*)(buffer.data() + sizeof(task)), thread_num);
				TaskScheduler::handleResult(buffer.data(), data);
			}

			while(true)
			{
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
				if(!flag)	break;
				int src = status.MPI_SOURCE;
				MPI_Get_count(&status, MPI_BYTE, &count);
				if(count > buffer.capacity())
					buffer.reserve(count);
				// cerr << "* Probe from " << src << endl;
				MPI_Recv(buffer.data(), count, MPI_BYTE, src, 0, MPI_COMM_WORLD, &status);
				// cerr << "* Recv task from " << src << endl;
				task = task_all.take(BATCH_SIZE);
				if(task.rest <= 0) rest--;
				MPI_Send(&task, sizeof(Task), MPI_BYTE, src, 0, MPI_COMM_WORLD);
				// cerr << "* Send task to " << src << " Rest = " << task_all.rest << endl;
				TaskScheduler::handleResult(buffer.data(), data);
			}
			
		}

		write(data, task_all.width, task_all.height, output);
		delete[] data;
	}
	else
	{
		while(true)
		{
			Task task = TaskScheduler::getNextTask();
			if(task.rest <= 0)
				break;
			int dataSize = task.rest * sizeof(int) + sizeof(task);
			if(dataSize > buffer.capacity())
				buffer.reserve(dataSize);
			*(Task*)buffer.data() = task;
			work_on_threads(task, (int*)(buffer.data() + sizeof(task)), thread_num);
			TaskScheduler::sendTaskResult(buffer.data(), dataSize);
		}
	}
	// cerr << id << " end." << endl;
}

void staticMPI (Task task_all, int thread_num, const char* output)
{
	const int n = task_all.rest;
	int n_proc = n / size + 1;
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
			Task task = tasks_proc[i];
			int j = 0;
			do {
				data_all[task.idInSuper] = datai[j++];
			}
			while(task.goNext());
		}
		write(data_all, task_all.width, task_all.height, output);
		delete[] buffer;
		delete[] data_all;
		cerr << clock() << " us: Process 0 Write end." << endl;
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
	
	auto task_all = Task {cf64(l, d), (r-l) / x_total, (u-d) / y_total, x_total, y_total, 1, 0, 0, 0, 1, x_total * y_total};
	
	// staticMPI(task_all, thread_num, output);
	dynamicMPI(task_all, thread_num, output);
	MPI_Finalize();
	return 0;
}
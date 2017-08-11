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

int calcIter (cf64 c)
{
	if(c.real()*c.real() + c.imag()*c.imag() < 1.0 / 16)
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
	u32 rest;

	bool goNext ()
	{
		move(step);
		idInSuper += stepInSuper;
		rest--;
		return rest > 0 && i < height;
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
};

class TaskScheduler
{
	static const int MASTER_RANK = 0;

	std::mutex mutex;
	bool readyGet0 = false;
	bool finish0 = false;
	Task task0;
	vector<char> buffer0;

	void handleResult (const char* buffer)
	{
		Task& task = *(Task*)buffer;
		auto bufint = (const int*)(buffer + sizeof(Task));
		do {
			data[task.idInSuper] = *(bufint++);
		}
		while(task.goNext());
	}

public:

	void scheduler (Task task, int n)
	{
		data = new int[task.rest];
		auto tasks = task.splitN_Seq(n);
		auto emptyTask = Task{0};

		vector<char> buffer;

		int i = 0, rest = 0;
		// cerr << "* begin send all." << endl;

		// mutex.lock();
		// task0 = tasks[i++], rest += 1;
		// readyGet0 = true;
		// mutex.unlock();

		for(int j=1; j<size; ++j)
		{
			MPI_Send(&tasks[i++], sizeof(Task), MPI_BYTE, j, 0, MPI_COMM_WORLD);
			rest += 1;
		}
			
		// cerr << "* begin listening." << endl;
		int count;
		MPI_Request request;
		MPI_Status status;
		int flag;
		bool first = true;
		for(; rest; first = false)
		{
			// if(finish0 && mutex.try_lock())
			// {
			// 	handleResult(buffer0.data());
			// 	// cerr << "* Recv task from 0" << endl;
			// 	if(i >= tasks.size()) rest--;
			// 	task0 = i < tasks.size()? tasks[i++]: emptyTask;
			// 	finish0 = false;
			// 	readyGet0 = true;
			// 	mutex.unlock();
			// 	// cerr << "* Send task " << (i-1) << " to " << 0 << endl;
			// 	continue;
			// }
			// Prepare for recv
			MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			int src = status.MPI_SOURCE;
			MPI_Get_count(&status, MPI_BYTE, &count);
			buffer.reserve(count);
			// cerr << "* Probe from " << src << endl;
			// Recv a result
			MPI_Recv(buffer.data(), count, MPI_BYTE, src, 0, MPI_COMM_WORLD, &status);
			// cerr << "* Recv task from " << src << endl;
			// Send new task
			if(i >= tasks.size()) rest--;
			const Task& task = i < tasks.size()? tasks[i++]: emptyTask;
			MPI_Isend(&task, sizeof(Task), MPI_BYTE, src, 0, MPI_COMM_WORLD, &request);
			// cerr << "* Send task " << (i-1) << " to " << src << endl;
			// Handle result
			handleResult(buffer.data());
		}
	}

	int* data;
	Task getNextTask ()
	{
		// cerr << id << " try to get next." << endl;
		if(id == MASTER_RANK)
		{
			while(!readyGet0)
				usleep(500);
			mutex.lock();
			Task task = task0;
			readyGet0 = false;
			mutex.unlock();
			return task;
		}
		Task task;
		MPI_Status status;
		MPI_Recv(&task, sizeof(Task), MPI_BYTE, MASTER_RANK, 0, MPI_COMM_WORLD, &status);
		// cerr << id << " get task:" << task.i << ' ' << task.j << endl;
		return task;
	}

	char* buffer = nullptr;
	MPI_Request sendRequest;
	void beginSendTaskResult (Task task, const int* data)
	{
		// cerr << id << " try to send result." << endl;
		int count = task.rest;
		int bufferSize = sizeof(Task) + count * sizeof(int);
		if(id == MASTER_RANK)
		{
			while(finish0 || readyGet0);
			mutex.lock();
			buffer0.reserve(bufferSize);
			memcpy(buffer0.data(), &task, sizeof(task));
			memcpy(buffer0.data() + sizeof(Task), data, count * sizeof(int));
			finish0 = true;
			mutex.unlock();
			return;
		}
		buffer = new char[bufferSize];
		memcpy(buffer, &task, sizeof(task));
		memcpy(buffer + sizeof(Task), data, count * sizeof(int));
		MPI_Isend(buffer, bufferSize, MPI_BYTE, MASTER_RANK, 0, MPI_COMM_WORLD, &sendRequest);
	}
	void endSendTaskResult ()
	{
		if(id == MASTER_RANK)
			return;
		MPI_Status status;
		MPI_Wait(&sendRequest, &status);
		delete[] buffer;
	}
	void sendTaskResult (Task task, const int* data)
	{
		beginSendTaskResult(task, data);
		endSendTaskResult();
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
	// omp_set_num_threads(thread_num);
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
	TaskScheduler ts;
	omp_set_nested (true);
	assert(omp_get_nested() == true);
	#pragma omp parallel
	{
		if(id == 0)
			#pragma omp single
			{
				ts.scheduler(task_all, 128);
			}
		else
		#pragma omp single 
		{
			vector<int> data;
			while(true)
			{
				Task task = ts.getNextTask();
				if(task.rest == 0)
					break;
				data.reserve(task.rest);
				work_on_threads(task, data.data(), thread_num);
				ts.sendTaskResult(task, data.data());
			}
			
			cerr << id << " end." << endl;
		}
		if(id == 0)
		{
			write(ts.data, task_all.width, task_all.height, output);
		}
	}
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
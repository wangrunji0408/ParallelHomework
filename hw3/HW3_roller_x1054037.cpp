#include <iostream>
#include <queue>
#include <pthread.h>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <chrono>
using namespace std;
using namespace std::chrono;

const int MAX_THREAD_NUM = 15;
const int MAX_C = 10;
int n, C, T, N;
high_resolution_clock::time_point t0;
std::stringstream fout;
std::ofstream ffout;

bool inQueue[MAX_THREAD_NUM];
pthread_mutex_t peopleMutex[MAX_THREAD_NUM];
pthread_mutex_t foutMutex;

class AQueue
{
	std::queue<int> que;
	pthread_mutex_t mutex;
public:
	AQueue()
	{
		pthread_mutex_init(&mutex, NULL);
	}
	~AQueue()
	{
		pthread_mutex_destroy(&mutex);
	}
	void push (int x)
	{
		pthread_mutex_lock(&mutex);
		que.push(x);
		inQueue[x] = true;
		pthread_mutex_unlock(&mutex);
	}
	void popN (int n, int* buf)
	{
		pthread_mutex_lock(&mutex);
		for(int i=0; i<n; ++i)
			buf[i] = que.front(), que.pop(), inQueue[buf[i]] = false;
		pthread_mutex_unlock(&mutex);
	}
	int size () const
	{
		return que.size();
	}
} que;

int getClockMs() 
{
	return duration_cast<milliseconds>(high_resolution_clock::now() - t0).count();
}

int getClockUs() 
{
	return duration_cast<microseconds>(high_resolution_clock::now() - t0).count();
}

inline void sleepUntil(int t)
{
	usleep(t * 1000 - getClockUs());
}

void* car_thread (void* p)
{
	int seat[MAX_C];
	for(int i=0; i<N; ++i)
	{
		while(que.size() < C);
		int beginMs = getClockMs();
		que.popN(C, seat);

		pthread_mutex_lock(&foutMutex);
		fout << "Car departures at " << beginMs << " millisec. Passenger ";
		for(int j=0; j<C; ++j)
			fout << seat[j] << ' ';
		fout << "are in the car.\n";
		pthread_mutex_unlock(&foutMutex);
		
		// usleep(T * 1000);
		int targetMs = beginMs + T;
		sleepUntil(targetMs);

		pthread_mutex_lock(&foutMutex);
		fout << "Car arrives at " << targetMs << " millisec. Passenger ";
		for(int j=0; j<C; ++j)
			fout << seat[j] << ' ';
		fout << "get off the car.\n";
		pthread_mutex_unlock(&foutMutex);

		for(int j=0; j<C; ++j)
			pthread_mutex_unlock(&peopleMutex[seat[j]]);
	}
	return NULL;
}

void* people_thread (void* p)
{
	int id = *(int*)p;
	while(true)
	{
		pthread_mutex_lock(&peopleMutex[id]);
		int beginMs = getClockMs();

		pthread_mutex_lock(&foutMutex);
		fout << "Passenger " << id << " wanders around the park.\n";
		pthread_mutex_unlock(&foutMutex);

		// usleep(id * 1000);
		int targetMs = beginMs + id;
		sleepUntil(targetMs);

		// int t = getClockMs();
		pthread_mutex_lock(&foutMutex);
		fout << "Passenger " << id << " returns for another ride at " << targetMs << " millisec.\n";
		pthread_mutex_unlock(&foutMutex);
		
		que.push(id);
	}
}

int main (int argc, char* argv[])
{
	n = atoi(argv[1]);
	C = atoi(argv[2]);
	T = atoi(argv[3]);
	N = atoi(argv[4]);

	for(int i=0; i<=n; ++i)
		pthread_mutex_init(&peopleMutex[i], NULL);
	pthread_mutex_init(&foutMutex, NULL);

	t0 = std::chrono::high_resolution_clock::now();

	pthread_t thread[MAX_THREAD_NUM];
	pthread_create(&thread[0], NULL, car_thread, NULL);
	for(int i=1; i<=n; ++i)
		pthread_create(&thread[i], NULL, people_thread, new int(i));
	pthread_join(thread[0], NULL);

	ffout.open(argv[5]);
	ffout << n << ' ' << C << ' ' << T << ' ' << N << endl;
	ffout << fout.str();
	return 0;
}
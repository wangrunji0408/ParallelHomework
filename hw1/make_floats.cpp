#include <iostream>
#include <cstdlib>
using namespace std;

inline float rand01 ()
{
	return (float)rand() / RAND_MAX;
}

int main (int argv, char* argc[])
{
	srand(time(0));
	int n = atoi(argc[1]);
	for(int i=0; i<n; ++i)
	{
		float f = rand01();
		cout.write((const char*)&f, sizeof(float));
	}
	return 0;
}
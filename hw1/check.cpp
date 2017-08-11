#include <fstream>
#include <iostream>
#include <cstdlib>
#include <limits>
using namespace std;

int main (int argv, char* argc[])
{
	ifstream fin(argc[1]);
	float last = -1e38; 
	float f;
	for(int i = 0; !fin.eof(); ++i)
	{
		fin.read((char*)&f, sizeof(float));
		if(f < last)
		{
			cout << "Wrong at " << i << endl;
			return 0;
		}
		last = f;
	}
	return 0;
}

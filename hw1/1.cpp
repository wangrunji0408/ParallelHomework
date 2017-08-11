#include <iostream>
#include <algorithm>
#include <fstream>
#include <vector>
#include <mpi.h>
using namespace std;

int main (int argc, char* argv[])
{
	int n = atoi(argv[1]);
	const char *input = argv[2];
	const char *output = argv[3];
	FILE* fin = fopen(input, "rb");
	FILE* fout = fopen(output, "wb");
	float *data = new float[n];
	fread(data, sizeof(float), n, fin);
	sort(data, data + n);
	fwrite(data, sizeof(float), n, fout);
}

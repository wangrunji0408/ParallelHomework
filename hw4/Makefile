CC                      := gcc
CXX                     := g++
CCFLAGS                 := -O3 -march=native -Wall -std=gnu11
CXXFLAGS                := -O3 -march=native -Wall -std=c++0x

NVCC                    := nvcc
NVFLAGS                 := -arch=sm_35 -O3

OMPFLAGS                := -fopenmp

EXES                    := block_FW.exe seq_FW.exe HW4_cuda.exe HW4_openmp.exe

all: $(EXES)

clean:
	rm -f $(EXES)

block_FW.exe: block_FW.cpp
	$(CXX) $(CXXFLAGS) -o $@ $?

seq_FW.exe: seq_FW.cpp
	$(CXX) $(CXXFLAGS) -o $@ $?

HW4_cuda.exe: HW4_x1054037_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $?

HW4_openmp.exe: HW4_x1054037_openmp.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(OMPFLAGS)" -o $@ $?

CC			:= gcc
CXX			:= g++

LDFLAGS			:= -lpthread -lncurses -std=c++11

all: roller frogger

roller: HW3_roller_x1054037.cpp
	$(CXX) $(LDFLAGS) -o $@ $?
frogger: HW3_frogger_x1054037.cpp
	$(CXX) $(LDFLAGS) -o $@ $?
clean:
	rm -f roller frogger

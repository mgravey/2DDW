export PWD = $(shell pwd)
export HEADER = $(PWD)/../include
export INC=-I/opt/local/include -I/usr/include
export SRC = $(PWD)/../src
export CC=icc
export CXX=icpc
export NVCC=nvcc

export CFLAGS=-O3 -std=c++11 -I$(HEADER)
export CXXFLAGS=-xhost -fopenmp

export WITH_OPENCV=-DWITH_OPENCV `pkg-config --cflags --libs opencv`

all: run

%.o: $(SRC)/%.cpp 
	$(CXX) -c -o $@ $<  $(CFLAGS) $(CXXFLAGS) $(INC) $(WITH_OPENCV)

2ddw:main.cpp 2ddw.o
	$(CXX) main.cpp 2ddw.o -std=c++11 -o 2ddw $(CFLAGS) $(CXXFLAGS) $(INC) $(WITH_OPENCV)

run:2ddw
	./2ddw 000.png 001.png

clean:
	@rm -rf *.o
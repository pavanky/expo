# replace the path to the appropriate location
CUDADIR := /usr/local/cuda

#Comment this line if using 32 bit
OS=64

NVCC = $(CUDADIR)/bin/nvcc
LDFLAGS += -L$(CUDADIR)/lib$(OS)
LDFLAGS += -lcuda -lcudart
CFLAGS += -I$(CUDADIR)/include

BIN = knn gauss

all : $(BIN)

%.o : %.cu
	$(NVCC) $(CFLAGS) -c $<

clean :
	rm -f *.o $(BIN)


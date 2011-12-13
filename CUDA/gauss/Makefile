include ../Makefile.common

gauss : gaussElim.cu
	$(CUDADIR)/bin/nvcc -o gauss gaussElim.cu -I$(CUDADIR)/include -L$(CUDADIR)/lib$(OS) -lcuda -lcudart

clean :
	rm -f gauss


include ../Makefile.common

knn :
	$(CUDADIR)/bin/nvcc -o knn knn.cu -I$(CUDADIR)/include -L$(CUDADIR)/lib$(OS) -lcuda -lcudart

clean :
	rm -f knn


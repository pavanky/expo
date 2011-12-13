SRC=matrix_multiply.c

all : mmul mmul_O3 mmul_O3_OMP

mmul : $(SRC)
	gcc $(SRC) -std=c99 -o mmul

mmul_O3 : $(SRC)
	gcc $(SRC) -std=c99 -o mmul_O3 -O3

mmul_O3_OMP : $(SRC)
	gcc $(SRC) -std=c99 -o mmul_O3_OMP -O3 -fopenmp

clean :
	rm -f mmul mmul_O3 mmul_O3_OMP

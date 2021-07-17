CC       = gcc
CFLAGS   = -fopenmp -O3 -Wall # -fPIC
LINKER  = gcc
LFLAGS = -fopenmp -lm -lblas

default: libconvGemm.so

libconvGemm.so: convGemm.o gemm_blis.o gemm_nhwc.o im2row_nhwc.o
	$(LINKER) $(LFLAGS) -shared -o $@ $^

%.o: %.c *.h
	$(CC) $(CFLAGS) -c $<

tags: *.c *.h
	ctags $^

#-----------------------------------

clean:
	rm *.o libconvGemm.so

#-----------------------------------


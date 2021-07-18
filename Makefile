CC       = gcc
CFLAGS   = -fopenmp -O3 -Wall # -fPIC
LINKER  = gcc
LFLAGS = -fopenmp -lm -lblas

default: libconvGemm.so test

libconvGemm.so: convGemm.o gemm_blis.o gemm_nhwc.o im2row_nhwc.o
	$(LINKER) $(LFLAGS) -shared -o $@ $^


runtest: test test.dat
	rm test.out
	while read line; do echo $$line | tee -a test.out; ./test $$line | tee -a test.out; done < test.dat

test.dat: test.pl test.txt
	perl $^ > $@

test: test.o convGemm.o gemm_blis.o gemm_nhwc.o im2row_nhwc.o
	$(LINKER) $(LFLAGS) -o $@ $^

%.o: %.c *.h
	$(CC) $(CFLAGS) -c $<

tags: *.c *.h
	ctags $^

#-----------------------------------

clean:
	rm *.o libconvGemm.so

#-----------------------------------


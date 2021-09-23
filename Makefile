BLIS_DIR = /home/andres/github/blis
BLIS_ARCH = skx
BLIS_INC = $(BLIS_DIR)/include/$(BLIS_ARCH)
BLIS_LIB = $(BLIS_DIR)/lib/$(BLIS_ARCH)

CC     = gcc
CFLAGS = -fopenmp -O3 -I$(BLIS_INC) # -fPIC -Wall
LINKER = gcc
LFLAGS = -fopenmp -lm -lblis -L$(BLIS_LIB) -Wl,-rpath=$(BLIS_LIB)

default: libconvGemm.so test

libconvGemm.so: convGemm.o gemm_blis.o gemm_nhwc.o im2row_nhwc.o
	$(LINKER) $(LFLAGS) -shared -o $@ $^


runtest: test test.dat
	rm -f test.out
	while read line; do echo -n $$line | tee -a test.out; ./test $$line | tee -a test.out; done < test.dat

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


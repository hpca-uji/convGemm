BLIS_DIR = /home/tomasan/blis
BLIS_ARCH = skx
BLIS_INC = $(BLIS_DIR)/include/$(BLIS_ARCH)
BLIS_LIB = $(BLIS_DIR)/lib/$(BLIS_ARCH)

CC     = gcc
CFLAGS = -fPIC -fopenmp -O3 -I$(BLIS_INC) # -Wall
LINKER = gcc
LFLAGS = -fPIC -fopenmp -lm -L$(BLIS_LIB) -Wl,-rpath=$(BLIS_LIB) -lblis

default: libconvGemm.so test test_gemm

libconvGemm.so: convGemm.o gemm_blis.o gemm_nhwc.o im2row_nhwc.o
	$(LINKER) -shared -o $@ $^ $(LFLAGS)


runtest: test test.dat
	rm -f test.out
	while read line; do echo -n $$line | tee -a test.out; ./test $$line | tee -a test.out; done < test.dat

test.dat: test.pl test.txt
	perl $^ > $@

test: test.o convGemm.o gemm_blis.o gemm_nhwc.o im2row_nhwc.o
	$(LINKER) -o $@ $^ $(LFLAGS)

test_gemm: test_gemm.o gemm_blis.o
	$(LINKER) -o $@ $^ $(LFLAGS)

%.o: %.c *.h
	$(CC) $(CFLAGS) -c $<

tags: *.c *.h
	ctags $^

#-----------------------------------

clean:
	rm *.o libconvGemm.so

#-----------------------------------


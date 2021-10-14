BLIS_DIR = $(HOME)/github/blis
BLIS_ARCH = skx
BLIS_INC = $(BLIS_DIR)/include/$(BLIS_ARCH)
BLIS_LIB = $(BLIS_DIR)/lib/$(BLIS_ARCH)

CC     = gcc
CFLAGS = -fopenmp -O3 -I$(BLIS_INC) # -Wall -DBENCHMARK
LINKER = gcc
LFLAGS = -fopenmp -lm -L$(BLIS_LIB) -Wl,-rpath=$(BLIS_LIB) -lblis

default: libconvGemm.so test test_trans test_back test_gemm

libconvGemm.so: convGemm.o gemm_blis.o gemm_nhwc.o gemm_back_nhwc.o im2row_nhwc.o gemm_nchw.o gemm_back_nchw.o im2col_nchw.o
	$(LINKER) -shared -o $@ $^ $(LFLAGS)

runtest: test test.dat
	rm -f test.out
	while read line; do ./test $$line | tee -a test.out; done < test.dat

runtrans: test_trans test.dat
	rm -f test_trans.out
	while read line; do ./test_trans $$line | tee -a test_trans.out; done < test.dat

runback: test_back test.dat
	rm -f test_back.out
	while read line; do ./test_back $$line | tee -a test_back.out; done < test.dat

test.dat: test.pl test.txt
	perl $^ > $@

test: test.o gemm_blis.o gemm_nhwc.o im2row_nhwc.o gemm_nchw.o im2col_nchw.o
	$(LINKER) -o $@ $^ $(LFLAGS)

test_trans: test_trans.o gemm_blis.o gemm_nhwc.o im2row_nhwc.o gemm_nchw.o im2col_nchw.o
	$(LINKER) -o $@ $^ $(LFLAGS)

test_back: test_back.o gemm_blis.o gemm_back_nhwc.o im2row_nhwc.o gemm_back_nchw.o im2col_nchw.o
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


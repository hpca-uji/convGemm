BLIS_DIR = $(HOME)/github/blis
BLIS_ARCH = skx
BLIS_INC = $(BLIS_DIR)/include/$(BLIS_ARCH)
BLIS_LIB = $(BLIS_DIR)/lib/$(BLIS_ARCH)

CC     = gcc
CFLAGS = -fopenmp -O3 -I$(BLIS_INC) # -Wall -DBENCHMARK
LINKER = gcc
LFLAGS = -fopenmp -lm -L$(BLIS_LIB) -Wl,-rpath=$(BLIS_LIB) -lblis

default: libconvGemm.so test test_trans test_back test_gemm

libconvGemm.so: convGemm.o gemm_blis.o gemm_blis_B3A2C0.o gemm_blis_A3B2C0.o im2row_nhwc.o im2col_nchw.o
	$(LINKER) -shared -o $@ $^ $(LFLAGS)

alltest: runtest runtrans runback

runtest: test test.in
	rm -f test.out
	while read line; do echo $$line; ./test $$line >> test.out || break; done < test.in

runtrans: test_trans test.in
	rm -f test_trans.out
	while read line; do echo $$line; ./test_trans $$line >> test_trans.out || break; done < test.in

runback: test_back test.in
	rm -f test_back.out
	while read line; do echo $$line; ./test_back $$line >> test_back.out || break; done < test.in

test.in: test.pl test.txt
	perl $^ > $@

test: test.o gemm_blis.o gemm_blis_B3A2C0.o gemm_blis_A3B2C0.o im2row_nhwc.o im2col_nchw.o
	$(LINKER) -o $@ $^ $(LFLAGS)

test_trans: test_trans.o gemm_blis.o gemm_blis_B3A2C0.o gemm_blis_A3B2C0.o im2row_nhwc.o im2col_nchw.o
	$(LINKER) -o $@ $^ $(LFLAGS)

test_back: test_back.o gemm_blis.o gemm_blis_B3A2C0.o gemm_blis_A3B2C0.o im2row_nhwc.o im2col_nchw.o
	$(LINKER) -o $@ $^ $(LFLAGS)

test_gemm: test_gemm.o gemm_blis.o gemm_blis_B3A2C0.o gemm_blis_A3B2C0.o
	$(LINKER) -o $@ $^ $(LFLAGS)

%.o: %.c *.h
	$(CC) $(CFLAGS) -c $<

tags: *.c *.h
	ctags $^

#-----------------------------------

clean:
	rm *.o *.out *.in libconvGemm.so

#-----------------------------------


BLIS_DIR = $(HOME)/github/blis
BLIS_ARCH = skx
BLIS_INC = $(BLIS_DIR)/include/$(BLIS_ARCH)
BLIS_LIB = $(BLIS_DIR)/lib/$(BLIS_ARCH)

MKL_LIB = -L${MKLROOT}/lib/intel64 -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread

CC     = gcc
CFLAGS = -fopenmp -O3 -I$(BLIS_INC) # -Wall -DBENCHMARK
LINKER = gcc
LFLAGS = -fopenmp -lm -L$(BLIS_LIB) -Wl,-rpath=$(BLIS_LIB) -lblis $(MKL_LIB)

default: libconvGemm.so test test_trans test_back test_gemm

libconvGemm.so: convGemm.o gemm_blis.o gemm_blis_B3A2C0.o gemm_blis_A3B2C0.o im2row_nhwc.o im2col_nchw.o
	$(LINKER) -shared -o $@ $^ $(LFLAGS)

alltest: runtest runtrans runback

rungemm: test_gemm
	rm -fr test_gemm.out
	for i in `seq 1000 1000 10000`; do for j in 1 2 3 4 5; do ./test_gemm $$i $$i $$i 10 >> test_gemm.out; done; done

runtest: test test.in
	rm -f test.out
	while read line; do echo $$line; for j in 1 2 3 4 5; do ./test 6 $$line >> test.out || break; done; done < test.in

runtrans: test_trans test.in
	rm -f test_trans.out
	while read line; do echo $$line; for j in 1 2 3 4 5; do ./test_trans 6 $$line >> test_trans.out || break; done ; done < test.in

runback: test_back test.in
	rm -f test_back.out
	while read line; do echo $$line; for j in 1 2 3 4 5; do ./test_back 6 $$line >> test_back.out || break; done ; done < test.in

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


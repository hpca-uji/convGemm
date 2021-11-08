all: convgemm-volta.pdf convgemm_nchw-volta.pdf trans-volta.pdf trans_nchw-volta.pdf back-volta.pdf back_nchw-volta.pdf blas-volta.pdf blis-volta.pdf

blas-%.eps: test_gemm.plt test_gemm_mkl.dat test_gemm_blis.dat test_gemm_cbuf_small.dat test_gemm_cbuf_small_A3.dat
	gnuplot -c $^ $* > $@

blis-%.eps: test_gemm2.plt test_gemm_blis.dat test_gemm_cbuf_small.dat test_gemm_cbuf_big.dat test_gemm_cbuf_big_seq.dat
	gnuplot -c $^ $* > $@

test_gemm_mkl.dat: test_gemm.pl test_gemm_cbuf_big.out test_gemm_cbuf_small.out test_gemm_cbuf_big_seq.out
	perl test_gemm.pl 9 test_gemm_cbuf_big.out test_gemm_cbuf_small.out  test_gemm_cbuf_big_seq.out> $@

test_gemm_blis.dat: test_gemm.pl test_gemm_cbuf_big.out test_gemm_cbuf_small.out test_gemm_cbuf_big_seq.out
	perl test_gemm.pl 10 test_gemm_cbuf_big.out test_gemm_cbuf_small.out  test_gemm_cbuf_big_seq.out> $@

test_gemm_cbuf_small.dat: test_gemm.pl test_gemm_cbuf_small.out
	perl test_gemm.pl 11 test_gemm_cbuf_small.out > $@

test_gemm_cbuf_small_A3.dat: test_gemm.pl test_gemm_cbuf_small.out
	perl test_gemm.pl 12 test_gemm_cbuf_small.out > $@

test_gemm_cbuf_big.dat: test_gemm.pl test_gemm_cbuf_big.out
	perl test_gemm.pl 11 test_gemm_cbuf_big.out > $@

test_gemm_cbuf_big_seq.dat: test_gemm.pl test_gemm_cbuf_big_seq.out
	perl test_gemm.pl 11 test_gemm_cbuf_big_seq.out > $@

convgemm-%.eps: convgemm.plt test.out
	gnuplot -c $^ $* > $@

convgemm_nchw-%.eps: convgemm_nchw.plt test.out
	gnuplot -c $^ $* > $@

trans-%.eps: trans.plt test_trans.out
	gnuplot -c $^ $* > $@

trans_nchw-%.eps: trans_nchw.plt test_trans.out
	gnuplot -c $^ $* > $@

back-%.eps: back.plt test_back.out
	gnuplot -c $^ $* > $@

back_nchw-%.eps: back_nchw.plt test_back.out
	gnuplot -c $^ $* > $@

%.pdf: %.eps
	epstopdf $<


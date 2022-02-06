convGemmNHWC
============

The convGemmNHWC library performs the convolution operation using an implicit im2row or im2col over a GEMM operation
with matrices in either the NHWC or NCHW format, respectively.


Compilation and installation
----------------------------

To compile and install the convGemmNHWC library, the Blis library should be installed on the system following the next
structure:

    BLIS_INSTALL_PREFIX
    |-- include/blis/blis.h
    `-- lib/libblis.so

Once the Blis library is installed, the next commands should be executed to compile and install the convGemmNHWC
library:

```shell
cd build
cmake [-D CMAKE_PREFIX_PATH=BLIS_INSTALL_PREFIX] [-D CMAKE_INSTALL_PREFIX=INSTALL_PREFIX] ..
make                 # Alternatively:  cmake --build . --clean-first
make install         # Alternatively:  cmake --install .' (this does not work with cmake older versions)
```

where ``BLIS_INSTALL_PREFIX`` is the prefix PATH where Blis is installed and ``INSTALL_PREFIX`` is the prefix PATH
where ``lib/libconvGemm.so`` and ``include/convgemm.h`` will be installed.

The ``-D CMAKE_PREFIX_PATH=BLIS_INSTALL_PREFIX`` option of the ``cmake ..`` command only is necessary if:

1. Blis is not installed in a system PATH,
2. the environment variable ``LD_LIBRARY_PATH`` is not defined or does not include the ``BLIS_INSTALL_PREFIX`` PATH, and
3. ``BLIS_INSTALL_PREFIX`` is different of ``INSTALL_PREFIX``.

As for the ``-D CMAKE_PREFIX_PATH=BLIS_INSTALL_PREFIX`` option, it is only required if the convGemmNHWC library should
be installed on a prefix PATH different of ``/usr/local``.

For example, if Blis is installed under ``~/opt/hpca_pydtnn`` and the convGemm library should be installed also under
that directory, the next commands should be executed:

```shell
cd build
cmake -D CMAKE_INSTALL_PREFIX=~/opt/hpca_pydtnn ..
make                 # Alternatively:  cmake --build . --clean-first
make install         # Alternatively:  cmake --install .' (this does not work with cmake older versions)
```


Running the tests
-----------------

To run the included tests, the next commands should be executed: 
```shell
cd build
cmake -D CMAKE_INSTALL_PREFIX=~/opt/hpca_pydtnn -D COMPILE_TESTS=ON ..
make TEST_TO_BE_RUN  # Alternatively:  cmake --build . --target=TEST_TO_BE_RUN
```

where ``TEST_TO_BE_RUN`` is one of the following:
- run_all_tests
- run_test_base
- run_test_trans
- run_test_back

The tests output files will be written to the ``build/tests/`` subdirectory.

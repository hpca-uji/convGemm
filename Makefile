CC       = gcc
CFLAGS   = -O3 -Wall # -fPIC
LINKER  = gcc
LFLAGS = -lm -lblas

default: libconvGemm.so

libconvGemm.so: convGemm.o
	$(LINKER) $(LFLAGS) -shared -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $<

#-----------------------------------

clean:
	rm *.o libconvGemm.so

#-----------------------------------


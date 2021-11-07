set term postscript color eps

set style data boxplot
set style boxplot # nooutliers

set title 'convGEMM back ' . ARG2
set ylabel 'Acceleration'
set xtics ("BLAS+row2im\nvs B3A2C0 NHWC" 1, "BLAS+row2im\nvs A3B2C0 NHWC" 2, "trans+col2im+BLAS\nvs B3A2C0 NCHW" 3, "trans+col2im+BLAS\nvs A3B2C0 NCHW" 4)
#, "convGEMM NCHW\nvs convGEMM NHWC" 3)
set key off
set bmargin 3

plot 1.0 with line ls 0, \
     ARG1 using (1):(($4+$5)/$6), \
     ARG1 using (2):(($4+$5)/$7), \
     ARG1 using (3):(($8+$9+$10)/$11), \
     ARG1 using (4):(($8+$9+$10)/$12)

set term postscript color eps

set style data boxplot
set style boxplot # nooutliers

set title 'convGEMM ' . ARG2
set ylabel 'Acceleration'
set xtics ("im2row+BLAS+bias\nvs B3A2C0 NHWC" 1, "im2row+BLAS+bias\nvs A3B2C0 NHWC" 2, "im2col+BLAS+trans+bias\nvs B3A2C0 NCHW" 3, "im2col+BLAS+trans+bias\nvs A3B2C0 NCHW" 4)
# , "convGEMM NCHW\nvs convGEMM NHWC" 3)
set key off
set bmargin 3

plot 1.0 with line ls 0, \
     ARG1 using (1):(($4+$5+$6)/$7), \
     ARG1 using (2):(($4+$5+$6)/$8), \
     ARG1 using (3):(($9+$10+$11)/$12), \
     ARG1 using (4):(($9+$10+$11)/$13)

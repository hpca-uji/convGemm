set term postscript color eps size 4,3
set encoding utf8

set object rectangle from graph 0,0 to graph 1,1 behind fillcolor rgb 'white' fillstyle solid noborder

set style data boxplot
set style boxplot nooutliers
# set style boxplot fraction 0.75
set boxwidth 100
set key left noautotitle
unset errorbars
set style fill transparent solid 0.25 border -1

set xtics 1000, 1000, 10000
set xlabel "Tamaño M=N=K"
set ylabel "GFlop/seg."

plot ARG1 using (  600):1 ls 1 title 'MKL', \
     ARG1 using ( 1600):2 ls 1, \
     ARG1 using ( 2600):3 ls 1, \
     ARG1 using ( 3600):4 ls 1, \
     ARG1 using ( 4600):5 ls 1, \
     ARG1 using ( 5600):6 ls 1, \
     ARG1 using ( 6600):7 ls 1, \
     ARG1 using ( 7600):8 ls 1, \
     ARG1 using ( 8600):9 ls 1, \
     ARG1 using ( 9600):10 ls 1, \
     ARG2 using (  800):1 ls 2 title 'BLIS', \
     ARG2 using ( 1800):2 ls 2, \
     ARG2 using ( 2800):3 ls 2, \
     ARG2 using ( 3800):4 ls 2, \
     ARG2 using ( 4800):5 ls 2, \
     ARG2 using ( 5800):6 ls 2, \
     ARG2 using ( 6800):7 ls 2, \
     ARG2 using ( 7800):8 ls 2, \
     ARG2 using ( 8800):9 ls 2, \
     ARG2 using ( 9800):10 ls 2, \
     ARG3 using ( 1000):1 ls 3 title 'B3A2C0', \
     ARG3 using ( 2000):2 ls 3, \
     ARG3 using ( 3000):3 ls 3, \
     ARG3 using ( 4000):4 ls 3, \
     ARG3 using ( 5000):5 ls 3, \
     ARG3 using ( 6000):6 ls 3, \
     ARG3 using ( 7000):7 ls 3, \
     ARG3 using ( 8000):8 ls 3, \
     ARG3 using ( 9000):9 ls 3, \
     ARG3 using (10000):10 ls 3, \
     ARG4 using ( 1200):1 ls 4 title 'A3B2C0', \
     ARG4 using ( 2200):2 ls 4, \
     ARG4 using ( 3200):3 ls 4, \
     ARG4 using ( 4200):4 ls 4, \
     ARG4 using ( 5200):5 ls 4, \
     ARG4 using ( 6200):6 ls 4, \
     ARG4 using ( 7200):7 ls 4, \
     ARG4 using ( 8200):8 ls 4, \
     ARG4 using ( 9200):9 ls 4, \
     ARG4 using (10200):10 ls 4

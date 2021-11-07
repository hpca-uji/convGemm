set term postscript color eps size 4,3
set object rectangle from graph 0,0 to graph 1,1 behind fillcolor rgb 'white' fillstyle solid noborder

set style data boxplot
set style boxplot sorted nooutliers
unset errorbars
set style fill transparent solid 0.25 border -1

set ylabel 'GFlop/seg.'
set xlabel "\n\nkn"
set key left
set yrange [0:1800]

set bmargin 4
set noxtics
set label    '3' at  1.25,0 offset 0,-1 rotate center font ",10"
set label   '16' at  2.25,0 offset 0,-1 rotate center font ",10"
set label   '24' at  3.25,0 offset 0,-1 rotate center font ",10"
set label   '32' at  4.25,0 offset 0,-1 rotate center font ",10"
set label   '48' at  5.25,0 offset 0,-1 rotate center font ",10"
set label   '64' at  6.25,0 offset 0,-1 rotate center font ",10"
set label   '96' at  7.25,0 offset 0,-1 rotate center font ",10"
set label  '112' at  8.25,0 offset 0,-1 rotate center font ",10"
set label  '128' at  9.25,0 offset 0,-1 rotate center font ",10"
set label  '144' at 10.25,0 offset 0,-1 rotate center font ",10"
set label  '160' at 11.25,0 offset 0,-1 rotate center font ",10"
set label  '192' at 12.25,0 offset 0,-1 rotate center font ",10"
set label  '224' at 13.25,0 offset 0,-1 rotate center font ",10"
set label  '256' at 14.25,0 offset 0,-1 rotate center font ",10"
set label  '288' at 15.25,0 offset 0,-1 rotate center font ",10"
set label  '320' at 16.25,0 offset 0,-1 rotate center font ",10"
set label  '352' at 17.25,0 offset 0,-1 rotate center font ",10"
set label  '384' at 18.25,0 offset 0,-1 rotate center font ",10"
set label  '416' at 19.25,0 offset 0,-1 rotate center font ",10"
set label  '448' at 20.25,0 offset 0,-1 rotate center font ",10"
set label  '480' at 21.25,0 offset 0,-1 rotate center font ",10"
set label  '512' at 22.25,0 offset 0,-1 rotate center font ",10"
set label  '528' at 23.25,0 offset 0,-1 rotate center font ",10"
set label  '544' at 24.25,0 offset 0,-1 rotate center font ",10"
set label  '576' at 25.25,0 offset 0,-1 rotate center font ",10"
set label  '608' at 26.25,0 offset 0,-1 rotate center font ",10"
set label  '640' at 27.25,0 offset 0,-1 rotate center font ",10"
set label  '672' at 28.25,0 offset 0,-1 rotate center font ",10"
set label  '704' at 29.25,0 offset 0,-1 rotate center font ",10"
set label  '736' at 30.25,0 offset 0,-1 rotate center font ",10"
set label  '768' at 31.25,0 offset 0,-1 rotate center font ",10"
set label  '800' at 32.25,0 offset 0,-1 rotate center font ",10"
set label  '832' at 33.25,0 offset 0,-1 rotate center font ",10"
set label  '864' at 34.25,0 offset 0,-1 rotate center font ",10"
set label  '896' at 35.25,0 offset 0,-1 rotate center font ",10"
set label  '928' at 36.25,0 offset 0,-1 rotate center font ",10"
set label  '960' at 37.25,0 offset 0,-1 rotate center font ",10"
set label  '992' at 38.25,0 offset 0,-1 rotate center font ",10"
set label '1024' at 39.25,0 offset 0,-1 rotate center font ",10"

plot ARG1 using (1):(2*$1*$2*$3*1e-9/($9+$10+$11)):(0.25):(sprintf("%6d", $1)) title 'im2col+BLIS+bias+trans', \
     ARG1 using (1.5):(2*$1*$2*$3*1e-9/$12):(0.25):(sprintf("%6d", $1)) title 'convGEMM NCHW'


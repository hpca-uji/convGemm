#!/usr/bin/perl -w

$col = shift;

while (<>) {
    @a = split;
    $k = $a[0];
    push(@{$data{$k}}, $a[$col]);
}

@keys = sort { $a <=> $b } keys %data;
$num_rows = 0;
for (@keys) {
    @{$data{$_}} = sort { $a <=> $b } @{$data{$_}};
    $n = @{$data{$_}};
    $num_rows = $n if $n > $num_rows;
}

print "# "; for (@keys) { print " $_"; } print "\n";

for ($i = 0; $i < $num_rows; $i++) {
    for (@keys) {
        print "$data{$_}[$i] ";
    }
    print "\n";
}

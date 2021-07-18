#!/usr/bin/perl -w

while(<>) {
    next if /^#/;
    chomp;
    ($h, $w, $c, $b, $kh, $kw, $kn) = split / /;
    $data{$_} = $h * $w * $c * $b * $kh * $kw * $kn;
}

for (sort { $data{$a} <=> $data{$b} } keys %data) {
    ($h, $w, $c, $b, $kh, $kw, $kn, $stride, $padding) = split / /;
    @args = ($b, $h, $w, $c, $kn, $kh, $kw, $padding, $padding, $stride, $stride);
    print "@args\n";
}

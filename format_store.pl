#!/usr/bin/env perl

perl -pe 's/\) \(/\)\n\(/g; s/[\(\)\.]//g;'  2 | perl -ne 'chomp; @x=split(/ +/); $_.="\n";$first=shift @x; @x=sort {$a <=> $b} @x; print("($first . (", join(" ", @x), "))"); END { print ")\n" } ' | perl -pe 's/\(#hash/#hash\(\(/'

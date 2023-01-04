chr="$1"

gnuplot -e "plot '/data/judong/ribosome-rna/hg001/chm13-aligmap/seqrun1.$chr.depth' using 2:3 with filledcurves y=0, '/data/judong/ribosome-rna/reference/chm13-geneplot-$chr-18s' using 1:2 with filledcurves y=0 fillstyle transparent solid 0.7, '/data/judong/ribosome-rna/reference/chm13-geneplot-$chr-28s' using 1:2 with filledcurves y=0 fillstyle transparent solid 0.7; pause mouse close"
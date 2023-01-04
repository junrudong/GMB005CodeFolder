run="$1"
reference="$2"

grch38_locus_list=$(cat /data/judong/RibosomalRnaGeneAlignment/Reference/grch38-genecoords | grep -E '^.+$' | awk '{print $2}')

samtools cat "$run".*.bam > "temp-$run.bam"
samtools index "temp-$run.bam"

samtools view "temp-$run.bam" $grch38_locus_list | awk '{print "@"$3":"$4"-"$1"\n"$10"\n+\n"$11}' | seqkit rmdup > "reads-$run.fq"

# these minimap2 parameters correspond the pbmm2 --preset=CCS
minimap2 -a -x asm20 -k 19 -O 5,56 -E 4,1 -B 5 -z 400,50 -r 2k --eqx --secondary=no -t 14 -K 2G "$reference" "reads-$run.fq" | samtools sort -o "$run.bam" 

samtools index "$run.bam"
for chr in $(seqkit seq --name "$reference"); do
    (
    samtools depth -a -r $chr "$run.bam" | awk '{if($2%1000==0){print $1,$2,s/1000;s=0}s+=$3}' > "$run.$chr.depth"
    echo $chr
    ) &
done
wait

rm "temp-$run.bam" "temp-$run.bam.bai"

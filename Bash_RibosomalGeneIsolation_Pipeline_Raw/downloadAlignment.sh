url="$1"
run="$2"

for range in $(cat /data/judong/ribosome-rna/reference/grch38-geneareas | grep -E '^.+$'); do
    (
    samtools view "$url" $range -o "$run.$range.bam"
    samtools index "$run.$range.bam"
    samtools depth -a "$run.$range.bam" | awk '{if($2%1000==0){print $1,$2,s/1000;s=0}s+=$3}' > "$run.$range.depth"
    echo $range
    ) &
done
wait
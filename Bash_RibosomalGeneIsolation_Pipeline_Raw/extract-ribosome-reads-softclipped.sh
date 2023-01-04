run="$1"

locus_list_18s_plus=$(cat /data/judong/Bash_RibosomalRnaGeneAlignment/Reference/chm13-genecoords | grep -E '^18S' | grep -E '\+$' | awk '{print $2}')
locus_list_18s_minus=$(cat /data/judong/Bash_RibosomalRnaGeneAlignment/Reference/chm13-genecoords | grep -E '^18S' | grep -E '\-$' | awk '{print $2}')
locus_list_28s_plus=$(cat /data/judong/Bash_RibosomalRnaGeneAlignment/Reference/chm13-genecoords | grep -E '^28S' | grep -E '\+$' | awk '{print $2}')
locus_list_28s_minus=$(cat /data/judong/Bash_RibosomalRnaGeneAlignment/Reference/chm13-genecoords | grep -E '^28S' | grep -E '\-$' | awk '{print $2}')

sam_to_fastq_softclipped () {
    awk '{
        sub(/^[^-]*-/, "", $1)

        if ($6 ~ "^[0-9]+S") {
            match($6, "^([0-9]+)S", a)

            $10=substr($10, a[1]+1)
            $11=substr($11, a[1]+1)
        }

        if ($6 ~ "[0-9]+S$") {
            match($6, "([0-9]+)S$", a)

            $10=substr($10, 1, length($10)-a[1])
            $11=substr($11, 1, length($11)-a[1])
        }

        print "@"$3":"$4"-"$1"\n"$10"\n+\n"$11
    }'
}

samtools view "$run.bam" $locus_list_18s_plus | sam_to_fastq_softclipped | seqkit rmdup > "18S/reads-$run.fq"
samtools view "$run.bam" $locus_list_18s_minus | sam_to_fastq_softclipped | seqkit rmdup | seqkit seq --reverse --complement >> "18S/reads-$run.fq"

samtools view "$run.bam" $locus_list_28s_plus | sam_to_fastq_softclipped | seqkit rmdup > "28S/reads-$run.fq"
samtools view "$run.bam" $locus_list_28s_minus | sam_to_fastq_softclipped | seqkit rmdup | seqkit seq --reverse --complement >> "28S/reads-$run.fq"

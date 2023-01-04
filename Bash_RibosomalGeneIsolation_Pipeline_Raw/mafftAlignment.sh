reference="$1"
filename="$2"

cat "$reference" "$filename" | seqkit seq --remove-gaps -w0 > "temp2-$filename"

mafft --auto --thread 16 "temp2-$filename" > "temp-$filename"

if [[ $? != 0 ]]; then echo "mafft error"; exit 1; fi
if [[ ! -e "temp-$filename" ]]; then echo "mafft no alignment"; exit 2; fi

seqkit seq -w0 "temp-$filename" > "$filename-aligned.fas"
rm "temp-$filename" "temp2-$filename"

alignment_region=$(seqkit head -n 1 "$filename-aligned.fas" | seqkit locate -P -r -p '[^-].*[^-]' --gtf | cut -f 4,5 | sed 's/\t/:/')
echo $alignment_region
seqkit subseq -r $alignment_region "$filename-aligned.fas" | seqkit grep --by-seq -P -r -p '[^-]' -w0 > "$filename-aligned-cut.fas"

if [[ $(seqkit seq --name "$filename-aligned-cut.fas" | wc -l) == 1 ]]; then echo "no alignment after cut"; exit 3; fi

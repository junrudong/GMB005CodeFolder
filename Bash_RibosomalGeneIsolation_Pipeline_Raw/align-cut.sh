reference="$1"
filename="$2"
overlap=$3

./pagan/bin/pagan2 -a "$reference" -q "$filename" --pileup --memory-for-single-alignment 10000 --force-gap --min-query-overlap $overlap --min-query-identity 0.7 --gap-extension 0.2 --end-gap-extension 0.999 -o "temp-$filename"

if [[ $? != 0 ]]; then echo "pagan error"; exit 1; fi
if [[ ! -e "temp-$filename.fas" ]]; then echo "pagan no alignment"; exit 2; fi

seqkit seq -w0 "temp-$filename.fas" > "$filename-aligned.fas"
rm "temp-$filename.fas"

alignment_region=$(seqkit head -n 1 "$filename-aligned.fas" | seqkit locate -P -r -p '[^-].*[^-]' --gtf | cut -f 4,5 | sed 's/\t/:/')

seqkit subseq -r $alignment_region "$filename-aligned.fas" | seqkit grep --by-seq -P -r -p '[^-]' -w0 > "$filename-aligned-cut.fas"

if [[ $(seqkit seq --name "$filename-aligned-cut.fas" | wc -l) == 1 ]]; then echo "no alignment after cut"; exit 3; fi
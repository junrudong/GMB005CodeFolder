reference="$1"
run="$2"

echo -n > "genes-$run.fas"
echo -n > "unaligned-$run"

for i in $(seq $(seqkit seq --name "reads-$run.fq" | wc -l)); do
    (
    seqkit range -r $i:$i -w0 "reads-$run.fq" > "read-$i-$run.fq"
    readname=$(seqkit seq --name "read-$i-$run.fq")
    #echo "$readname"

    ../../align-cut.sh "$reference" "read-$i-$run.fq" 0.1 >/dev/null
    retcode=$?

    if [[ $retcode != 0 ]]; then
        echo "$retcode $readname" >> "unaligned-$run"
    else
        seqkit range -r 2:2 -w0 "read-$i-$run.fq-aligned-cut.fas" > "read-$i-$run.fq-secondrun.fas"
        
        ../../align-cut.sh "$reference" "read-$i-$run.fq-secondrun.fas" 0.4 >/dev/null
        retcode=$?

        if [[ $retcode != 0 ]]; then
            echo "$(( $retcode + 3 )) $readname" >> "unaligned-$run"
        else
            seqkit range -r 2:2 -w0 "read-$i-$run.fq-secondrun.fas-aligned-cut.fas" >> "genes-$run.fas"
        fi
    fi

    rm "read-$i-$run.fq" "read-$i-$run.fq-aligned.fas" "read-$i-$run.fq-aligned.fas.seqkit.fai" "read-$i-$run.fq-aligned-cut.fas" "read-$i-$run.fq-secondrun.fas" "read-$i-$run.fq-secondrun.fas-aligned.fas" "read-$i-$run.fq-secondrun.fas-aligned.fas.seqkit.fai" "read-$i-$run.fq-secondrun.fas-aligned-cut.fas"
    ) &

    # only run 16 threads at a time
    if [[ $(jobs -r -p | wc -l) -ge 16 ]]; then wait -n; fi
done
wait

#!/bin/bash


echo "Started at $(date)";

start=`date +%s`

start_jobs=3264782
end_jobs=3264841

while [ $start_jobs -le $end_jobs ]
do
    echo $start_jobs
    scancel $start_jobs
    ((start_jobs++))
done

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
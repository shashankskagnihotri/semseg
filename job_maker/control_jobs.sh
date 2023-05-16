#!/bin/bash


echo "Started at $(date)";

start=`date +%s`

start_jobs=2198326
end_jobs=2198353

while [ $start_jobs -le $end_jobs ]
do
    echo $start_jobs
    scontrol update JOB=$start_jobs NICE=400
    ((start_jobs++))
done

end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime

# 2198328 2198332 2198336 2198340  2198344 2198348 2198352
#!/bin/sh
#created by PAUL GOYES
rm -f  gathers_CR.su
#######################################
#. ./utils/progress_bar.sh


start=$(date +%s)


nshots=50
for i in $(seq 1 1 $nshots)
do
   
   #echo "loading shot .... $i"
   shotload=shotsSU/gathers_$i.su
   
   sushw < $shotload key=fldr a=$i  | sunormalize norm=rms | suagc > shotsSU/temp_$i.su
   
   cat shotsSU/temp_$i.su >> gathers_CR.su
   #show_progress $i $nshots
   

done

end=$(date +%s)

echo "Elapsed Time: $(($end-$start)) seconds"

rm -f temp*

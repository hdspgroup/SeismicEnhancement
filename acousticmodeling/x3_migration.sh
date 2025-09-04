#!/bin/sh
#created by PAUL GOYES (goyes.yesid@gmail.com)
#date: 28/NOV/2023
#This work was support by grant 3925 funded by Universidad Industrial de Santander
#########################################################

####### VELOCITY MODEL PARAMETERS


rm -f velmig.bin

nx=100
dx=60
nz=100
dz=20

## ACQUISITION PARAMETERS
shotfile=shots.su

 #Required Parameters:							
nxo=$nx	   #number of total horizontal output samples			
nxshot=20	#number of shot gathers to be migrated								
vfile=V1kms	 #velocity profile, it must be binary format.
 



# remove upper 1 velocity grid lines corresponding to datum at 10 m depth
#NZ1=`expr $nz - 1`
#echo $NZ1

transp < $vfile n1=$nz > velmig.bin
 
 
 
 
 
### run the migration algorithm

start=$(date +%s)
sumigpresp <$shotfile nflag=1 nz=$nz dz=$dz  dx=$dx nxo=$nxo nxshot=$nxshot vfile=velmig.bin fmax=25 > temp
suagc < temp | sunormalize > poststack.su

rm -f temp

end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"

#Elapsed Time: 1333 seconds



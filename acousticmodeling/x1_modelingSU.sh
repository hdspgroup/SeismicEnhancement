#!/bin/sh
#created by PAUL GOYES (goyes.yesid@gmail.com)
#date: 28/NOV/2023
#This work was support by grant 3925 funded by Universidad Industrial de Santander
#########################################################

rm -f gathers.su

####### VELOCITY MODEL PARAMETERS

vmodel=Vunif3

nx=100
nz=100
dx=10
dz=10

## ACQUISITION PARAMETERS

xs0=0 #coordinate first shot in meters

dsx=20 #increment between shots in meters
nshots=50 #number of shots

TMAX=4 #seconds (total record time)

#IMPORTANT: check if $dsx*(nshots-1) + xs0$  less or equal $nx*dx$


###### FOLDER PREPARATION #############

#the directory "shotsSU" is created is does not exist
#all files inside "shotsSU" are deleted if folder exists
mkdir -p shotsSU
rm  -f shotsSU/*

###############################################################################
############## SEISMIC MODELING USING PARALLEL COMPUTING ######################
###############################################################################

# Function to perform the MODELING for each iteration
perform_calculation() {
  local iteration=$1
  xs=$(echo "$dsx*($i-1) + $xs0" | bc)
  echo "Simulating shot: $iteration at xs: $xs"
  # seismic wave modeling using finite differences 
  sufdmod2 < $vmodel nx=$nx nz=$nz xs=$xs zs=10 dx=$dx dz=$dz tmax=$TMAX fmax=25 hsz=10 hsfile=shotsSU/gathers_$i.su verbose=0 abs=0,1,1,1 > /dev/null
  echo " "
  echo "Done shot:: $iteration:"
}


for i in $(seq 1 1 $nshots)
do
   
   perform_calculation "$i" &
done

# Wait for all background processes to finish
wait
echo " "
echo "All Shots completed"
echo "All calculations completed"

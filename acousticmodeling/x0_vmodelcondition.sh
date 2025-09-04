#!/bin/sh
#created by PAUL GOYES

###only if necessary!!!!!!!!!!!!!!!!!!!!!!!!
#farith convert sloth to m/s

velbin=CR_m_s.bin 
nz=2000
nx=2000
dz=2.25
dx=8

#first: Smooth the model

resamp < $velbin  n1=$nz  n2=$nx  n1r=500 d1=$dz d1r=8 > velresam.bin
nz=500
dz=8

transp < velresam.bin n1=$nz > TEMPvel_nx_nz.bin

#pasamos de 2000 muestras a 1000, por lo que toca duplicar el dx... de 8 a 16
resamp < vel_nx_nz.bin  n1=$nx  n2=$nz  n1r=1000 d1=$dx d1r=16 > TEMPvel_nx_nzvelresam.bin

#ahora 
nx=1000
dx=16
#volvemos a transponer (y guardamos con el mismo nombre)
transp < TEMPvel_nx_nzvelresam.bin n1=$nx > velresam.bin 

smooth2 < velresam.bin  n1=$nz  n2=$nx r1=50 r2=1 > CR_m_s_smooth.bin 

###just plot in case:

ximage < CR_m_s_smooth.bin  n1=500 d1=$dz d2=$dx 

rm -f TEMP*

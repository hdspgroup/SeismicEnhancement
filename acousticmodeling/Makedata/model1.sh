#! /bin/sh
# File: model1.sh

# Set messages on
set -x

# Name output binary model file
modfile=model1.dat

# Name output encapsulated Postscript image file
psfile=modelo1.eps

# Remove previous .eps file
rm -f $psfile

trimodel xmin=0 xmax=6 zmin=0 zmax=2 \
1 xedge=0,6 \
  zedge=0,0 \
  sedge=0,0 \
2 xedge=0,2,4,6 \
  zedge=0.30,0.50,0.20,0.30 \
  sedge=0,0,0,0 \
3 xedge=0,2,4,6 \
  zedge=0.55,0.75,0.45,0.55 \
  sedge=0,0,0,0 \
4 xedge=0,2,4,6 \
  zedge=0.65,0.85,0.55,0.65 \
  sedge=0,0,0,0 \
5 xedge=0,2,4,6 \
  zedge=1.30,1.30,1.60,1.20 \
  sedge=0,0,0,0 \
6 xedge=0,6 \
  zedge=2,2 \
  sedge=0,0 \
 kedge=1,2,3,4,5,6 \
 sfill=0.1,0.1,0,0,0.44,0,0 \
 sfill=0.1,0.4,0,0,0.40,0,0 \
 sfill=0.1,0.6,0,0,0.35,0,0 \
 sfill=0.1,1.0,0,0,0.30,0,0 \
 sfill=0.1,1.5,0,0,0.20,0,0 > $modfile
##       x,z

# Create a Postscipt file of the model
#   Set gtri=1.0 to see sloth triangle edges
spsplot < $modfile > $psfile \
        gedge=0.5 gtri=2.0 gmin=0 gmax=1 \
        title="Modelo 1" \
        labelz="Profundidad (km)" labelx="Distancia (km)" \
        dxnum=1.0 dznum=0.5 wbox=6 hbox=2

# Exit politely from shell
exit


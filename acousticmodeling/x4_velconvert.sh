#! /bin/sh
# Conversion de modelos de velocidad (x,z) (v(z)), a vnmo(t), vrms(t) y v(t). 
#created by PAUL GOYES

#dt=727
dt=0.000727
nt=2063

nz=500
dz=8
nx=2000
dx=8

velconv intype=vintz outtype=vintt nt=$nt dt=$dt dz=$dz nz=$nz nx=$nx dx=$dx < CR_m_s_smooth.bin > velint.bin
ximage < velint.bin n1=$nt windowtitle='velint' legend=1 units="(m/s)" hbox=240 wbox=630 d1=$dt d2=$dx label1='time(s)' title='int-velocity' &

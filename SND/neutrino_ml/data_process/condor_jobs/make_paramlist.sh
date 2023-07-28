#!/bin/bash
MC_PATH="/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000"
PFILE=paramlist_neutrinoMC.txt
touch $PFILE
for f in $MC_PATH/* 
do 
	echo "neutrino ${f##*/}" >> $PFILE
done

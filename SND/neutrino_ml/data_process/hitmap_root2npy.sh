#!/bin/bash
MC_PATH="/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000"
for f in $MC_PATH/* 
do 
	echo ${f##*/}
	python hit_sandbox.py -mc $MC_PATH -t neutrino -p ${f##*/}
done

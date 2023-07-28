#!/bin/bash
SNDLHC_mymaster=/afs/cern.ch/user/g/golovati/private/
export ALIBUILD_WORK_DIR=$SNDLHC_mymaster/sw #for alienv

source /cvmfs/sndlhc.cern.ch/SNDLHC-2023/Jan22/setUp.sh
#source /cvmfs/sndlhc.cern.ch/SNDLHC-2022/July14/setUp.sh
echo $SNDSW_ROOT
echo 'loading alienv'
eval `alienv load --no-refresh sndsw/latest`
echo $SNDSW_ROOT

MC_PATH="/eos/experiment/sndlhc/MonteCarlo/Neutrinos/Genie/sndlhc_13TeV_down_volTarget_100fb-1_SNDG18_02a_01_000"
etype=$1
partition=$2

#4705 4654 4661 4713 4778 5113
OUTPUTDIR=/afs/cern.ch/user/g/golovati/work/neutrino_ml/MC_explore/data/

# File=$OUTPUTDIR/ds_detid_time_p${partition}_${runN}.npy # sndsw_raw-${partition}_${runN}_muonReco.root
# echo $File
# checks
#if [ $(stat -c%s "$File") -gt 0 ]
#then
#   return
#else
/cvmfs/sndlhc.cern.ch/SNDLHC-2023/Jan22/sw/slc7_x86-64/Python/v3.8.12-local1/bin/python $SNDLHC_mymaster/../work/neutrino_ml/MC_explore/condor_jobs/hit_sandbox.py -mc $MC_PATH -t $etype -p $partition -o $OUTPUTDIR
#fi

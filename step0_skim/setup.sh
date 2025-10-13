#!/bin/bash

if [ ! -d "CMSSW_15_0_7" ]; then
    echo "Setting up CMSSW_15_0_7"
    cmsrel CMSSW_15_0_7
fi

#Check if env var CMSSW_BASE is set
if [ -z "$CMSSW_BASE" ]; then
    cd CMSSW_15_0_7/src
    scramv1 runtime -sh
    cd ../..
fi

#Copy the Ntuplizer directory to CMSSW_15_0_7/src and overwrite if it exists
cp -rf Ntuplizer CMSSW_15_0_7/src/
cd CMSSW_15_0_7/src
scram b -j 8
cd ../..

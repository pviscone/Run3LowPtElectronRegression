#!/bin/bash
mkdir -p ../data
voms-proxy-init --rfc --voms cms -valid 192:00
dasgoclient -query="file dataset=/DoubleElectron_Pt-1To300_gun/Run3Winter24MiniAOD-FlatPU0to80_133X_mcRun3_2024_realistic_v8-v2/MINIAODSIM" > ../data/dataset2024.txt
dasgoclient -query="file dataset=/DoubleElectron_FlatPT-1to500_13p6TeV/Run3Summer23MiniAODv4-FlatPU0to70_130X_mcRun3_2023_realistic_v14-v2/MINIAODSIM" > ../data/dataset2023.txt
dasgoclient -query="file dataset=/DoubleElectron_FlatPT-1to500_13p6TeV/Run3Summer22MiniAODv3-FlatPU0to70_124X_mcRun3_2022_realistic_v12-v2/MINIAODSIM" > ../data/dataset2022.txt
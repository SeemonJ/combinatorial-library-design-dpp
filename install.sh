#!/bin/bash
# A script to setup all file structures and execute a full smaller run of the framework 
eval $(conda shell.bash hook)
set -e
conda env create -f environment.yml
git clone https://github.com/MolecularAI/Lib-INVENT
# This version of Lib-INVENT was hardcoded to always need a config.json in configurations
mv Lib-INVENT/configurations/example.config.json Lib-INVENT/configurations/config.json
conda env create -f aizynth-env.yml
# Everything needed to run AiZynthFinder should be 
# present in the environment except for the models
conda activate aizynth-env
download_public_data .
cd ..

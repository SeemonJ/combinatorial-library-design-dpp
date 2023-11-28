#!/bin/bash
# A script to setup all file structures and execute a full smaller run of the framework 
eval $(conda shell.bash hook)
set -e
# Running Lib-INVENT
conda activate comb-lib-design
cd Lib-INVENT
python input.py ../sample_run/libInvent_10_epochs.json
cd ..
# Extracting Synthons and BBs from Lib-INVENT output
python readLibInventOutput.py -i Lib-INVENT/sample_run.log/scaffold_memory.csv -n 1 -o sample_output

conda deactivate
# Running Retrosynthesis on BBs
conda activate aizynth-env
aizynthcli --smiles sample_output/AC_SMILES_split_0.smi --config sample_run/config.yml --output sample_output/aizynthfinder_output_AC.hdf5
aizynthcli --smiles sample_output/BH_SMILES_split_0.smi --config sample_run/config.yml --output sample_output/aizynthfinder_output_BH.hdf5
# Extracting Reaction availability from AiZynthFinder Output
python aggregateAizynthFinderOutput.py -i ./sample_output/aizynthfinder_output_AC.hdf5 -o ./sample_output/AC_output.csv -d ./sample_output/AC_Synthon_BB_dict.pkl
python aggregateAizynthFinderOutput.py -i ./sample_output/aizynthfinder_output_BH.hdf5 -o ./sample_output/BH_output.csv -d ./sample_output/BH_Synthon_BB_dict.pkl
conda deactivate
# Running Optimization
conda activate comb-lib-design
python main.py -r 5 -c 5 -s 20 -d 0 -ba sample_output/AC_output.csv -bb sample_output/BH_output.csv -da sample_output/AC_Synthon_BB_dict.pkl -db sample_output/BH_Synthon_BB_dict.pkl -o sample_output

# Plotting Results
python displayLibrary.py -ia sample_output_1/selected_rows.pkl -ib sample_output_1/selected_columns.pkl -da sample_run/AC_Synthon_BB_dict.pkl -db sample_run/BH_Synthon_BB_dict.pkl -o sample_selection.png
#!/bin/bash
# generate tar archive of required files
# 1. all py files, 2. all yaml files, 3. data/dataset_gap_Si-no_1_atom.pt 4. data/dataset_gap_Si-no_1_atom_indices.txt

# 1. gen file list from git
git ls-tree --full-tree --name-only -r HEAD > files.txt

# 2. append dataset files
echo "data/dataset_gap_Si-no_1_atom.pt" >> files.txt

# 3. generate tar archive
tar -czvf lightning.tar.gz -T files.txt

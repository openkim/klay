# ---------------------------------------------------------------------------------
echo "Generating local env for nequip/ML"
# ---------------------------------------------------------------------------------

mkdir -p bin
mkdir -p env
# get micromamba
# curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba

wget https://github.com/ipcamit/CECAM-ASE-2023/raw/master/micromamba_bin/linux.tar.gz
tar -xvf linux.tar.gz

export MAMBA_ROOT_PREFIX=`pwd`/env  # optional, defaults to ~/micromamba
eval "$(./bin/micromamba shell hook -s posix)"

micromamba activate
micromamba install python=3.10 -y -c conda-forge
#micromamba install pytorch==1.12.1  pytorch-cuda=11.7 -c pytorch -c nvidia
micromamba install pytorch==1.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

pip install torch_geometric

# Optional dependencies:
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu116.html


git clone https://github.com/mir-group/nequip
cd nequip
pip install -e .

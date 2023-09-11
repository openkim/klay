export MAMBA_ROOT_PREFIX=`pwd`/env  # optional, defaults to ~/micromamba
eval "$(./bin/micromamba shell hook -s posix)"
micromamba activate

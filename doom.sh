# Get submodules
git submodule update --init --recursive
# Setup Conda
conda create -n doom python=3.12
# Source env vars upon conda activation
mkdir -p $(conda info --base)/envs/doom/etc/conda/activate.d
echo "source $PWD/.env.base" > $(conda info --base)/envs/doom/etc/conda/activate.d/env_vars.sh
conda activate doom
# Build Image
docker build -t mujuni-image unitree_mujoco_container/.devcontainer/.

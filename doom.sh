# Get submodules
git submodule update --init --recursive
# Setup Conda
conda create -n doom python=3.x
conda activate doom
# Source env vars upon conda activation
mkdir -p $(conda info --base)/envs/doom/etc/conda/activate.d
echo "source .env.base" > $(conda info --base)/envs/doom/etc/conda/activate.d/env_vars.sh
# Build Image
docker build -t mujuni-image unitree_mujoco_container/.devcontainer/.

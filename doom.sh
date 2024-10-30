# Get submodules
git submodule update --init --recursive
# Setup Conda
conda create -n doom python=3.12
# Source env vars upon conda activation
mkdir -p $(conda info --base)/envs/doom/etc/conda/activate.d
echo -e "export DOOM_DIR=$PWD\nsource $PWD/.env.base" > $(conda info --base)/envs/doom/etc/conda/activate.d/env_vars.sh
conda activate doom
# Build Image
docker build -t mujuni-image unitree_mujoco_container/.devcontainer/.
# Setup Network Interface
sudo nmcli con add type ethernet ifname $NETWORK_INTERFACE ipv4.addresses 192.168.123.1/24 ipv4.method manual
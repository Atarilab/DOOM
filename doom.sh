#!/usr/bin/env bash

# Copyright (c) 2024, The ATARI-DOOM Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#==
# Configurations
#==

# Exits if error occurs
set -e

# Set tab-spaces
tabs 4

#==
# Helper functions
#==

# print the usage description
print_help () {
    echo -e "\nusage: $(basename "$0") [-h] [-i] -- Utility to manage ATARI DOOM."
    echo -e "\noptional arguments:"
    echo -e "\t-h, --help           Display the help content."
    echo -e "\t-i, --install        Install ATARI DOOM inside conda env, build docker container and setup network interface with Go2."
    echo -e "\t-b, --build          Build the docker image."
    echo -e "\n" >&2
}


# check argument provided
if [ -z "$*" ]; then
    echo "[Error] No arguments provided." >&2;
    print_help
    exit 1
fi

# Pass the arguments
while [[ $# -gt 0 ]]; do
    # Read the key
    case "$1" in
        -i|--install)

            # Initialize Conda for the current shell session
            eval "$(conda shell.bash hook)"
    
            # Get submodules
            git submodule update --init --recursive
            
            # Check if the Conda environment exists, and create it only if it doesn't
            if ! conda info --envs | grep -q "^doom\s"; then
                echo "Creating Conda environment 'doom'..."
                conda create -n doom python=3.12 -y
            else
                echo "Conda environment 'doom' already exists. Skipping creation."
            fi
    
            # Source env vars upon Conda activation
            mkdir -p $(conda info --base)/envs/doom/etc/conda/activate.d
    
            # Check if the file exists before creating it
            if [ ! -f "$(conda info --base)/envs/doom/etc/conda/activate.d/env_vars.sh" ]; then
                echo "Creating env_vars.sh to source environment variables..."
                echo -e "export DOOM_DIR=$PWD\nsource $PWD/.env.base" > $(conda info --base)/envs/doom/etc/conda/activate.d/env_vars.sh
            else
                echo "env_vars.sh already exists. Skipping file creation."
            fi
    
            # Activate the Conda environment
            conda activate doom
            
            # Build Docker Image if it doesn't exist
            if ! docker image inspect mujuni-image >/dev/null 2>&1; then
                echo "Building Docker image 'mujuni-image'..."
                docker build -t mujuni-image unitree_mujoco_container/.devcontainer/.
            else
                echo "Docker image 'mujuni-image' already exists. Skipping build."
            fi
            
            # Check if the network interface already exists, and add it only if it doesn't
            if ! nmcli con show | grep -q "$NETWORK_INTERFACE"; then
                echo "Adding network interface for $NETWORK_INTERFACE..."
                sudo nmcli con add type ethernet ifname $NETWORK_INTERFACE ipv4.addresses 192.168.123.1/24 ipv4.method manual
            else
                echo "Network interface '$NETWORK_INTERFACE' already exists. Skipping network setup."
            fi
    
    
            shift
            ;;

        -b|--build)
            # Build Docker Image if it doesn't exist
            if ! docker image inspect mujuni-image >/dev/null 2>&1; then
                echo "Building Docker image 'mujuni-image'..."
                docker build -t mujuni-image unitree_mujoco_container/.devcontainer/.
            else
                echo "Docker image 'mujuni-image' already exists. Skipping build."
            fi
            shift
            ;;

        -h|--help)
            print_help
            exit 1
            ;;
        
        *) # Unknown option
            echo "[Error] Invalid argument provided: $1"
            print_help
            exit 1
            ;;
    esac
done

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
    echo -e "\t-i, --install        Install ATARI DOOM, build docker container and setup network interface with Go2."
    echo -e "\t-b, --build          Build the docker image."
    echo -e "\t-e, --enter          Enter the DOOM docker container."
    echo -e "\t-d, --delete         Delete all existing DOOM docker containers and images."
    echo -e "\t-a, --attach         Attach shell to existing DOOM docker container."
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

            # Get submodules
            git submodule update --init --recursive
            
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
                docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t mujuni-image unitree_mujoco_container/.devcontainer/.
            else
                echo "Docker image 'mujuni-image' already exists. Skipping build."
            fi
            shift
            ;;

        -e|--enter)
            # Check if a container with the same name exists
            CONTAINER_NAME="DOOM"  # Replace with your container's desired name
            if ! docker ps -a --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
                # If container doesn't exist, create and start a new container
                xhost +local:root & \
                docker run -it --privileged \
                    --env-file .env.docker \
                    --network host \
                    --user $(id -u):$(id -g) \
                    --gpus all \
                    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
                    -v $PWD/src:/home/atari/workspace/DOOM/src \
                    --env XAUTHORITY=/root/.Xauthority \
                    -v $HOME/.Xauthority:/root/.Xauthority \
                    --name $CONTAINER_NAME mujuni-image
            else
                # If container exists, just start it
                echo "Container $CONTAINER_NAME already exists. Starting the container..."
                docker start -i $CONTAINER_NAME
            fi
            shift
            ;;


        -d|--delete)
            # Enter the docker container
            docker container prune -f
            docker image prune -f
            docker rmi mujuni-image
            shift
            ;;

        -a|--attach)
            # Attach shell to existing docker container
            docker exec -it $(docker ps --filter "ancestor=mujuni-image" -q | head -n 1) /bin/bash
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

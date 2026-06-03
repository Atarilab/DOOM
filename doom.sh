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


check_docker_access () {
    local docker_error

    if docker_error="$(docker info 2>&1 >/dev/null)"; then
        return 0
    fi

    echo "[Error] Docker is not accessible for user $USER." >&2
    echo "$docker_error" >&2

    if [ -S /var/run/docker.sock ] && ! id -nG | grep -qw docker; then
        echo "" >&2
        echo "Your user is not in the docker group. Run:" >&2
        echo "  sudo usermod -aG docker $USER" >&2
        echo "Then log out and back in, or run: newgrp docker" >&2
    else
        echo "" >&2
        echo "Check that the Docker daemon is running and that your user can access it." >&2
    fi

    exit 1
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

            check_docker_access
            
            # Build Docker Image if it doesn't exist
            if ! docker image inspect mujuni-image >/dev/null 2>&1; then
                echo "Building Docker image 'mujuni-image'..."
                docker build --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t mujuni-image unitree_mujoco_container/.devcontainer/.
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
            check_docker_access

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
            check_docker_access

            # Check if a container with the same name exists
            CONTAINER_NAME="DOOM"  # Replace with your container's desired name
            INPUT_GID="$(getent group input | cut -d: -f3)"
            if [ -z "$INPUT_GID" ]; then
                echo "[Error] Could not determine host input group ID." >&2
                echo "Make sure the input group exists on the host before starting the container." >&2
                exit 1
            fi

            if ! docker ps -a --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
                # If container doesn't exist, create and start a new container
                xhost +local:root & \
                docker run --shm-size=2g -it --privileged \
                    --env-file .env.docker \
                    --network host \
                    --device /dev/input \
                    --group-add "$INPUT_GID" \
                    --user $(id -u):$(id -g) \
                    --gpus all \
                    -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
                    -v /dev/shm:/dev/shm \
                    -v $PWD/src:/home/atari/workspace/DOOM/src \
                    -v $HOME/.Xauthority:/root/.Xauthority \
                    -v $PWD/.vscode:/home/atari/workspace/.vscode \
                    -v $PWD/pyproject.toml:/home/atari/workspace/pyproject.toml \
                    --env XAUTHORITY=/root/.Xauthority \
                    --name $CONTAINER_NAME mujuni-image
            else
                # If container exists, just start it
                echo "Container $CONTAINER_NAME already exists. Starting the container..."
                docker start -i $CONTAINER_NAME
            fi
            shift
            ;;


        -d|--delete)
            check_docker_access

            # Enter the docker container
            docker container prune -f
            docker image prune -f
            docker rmi mujuni-image
            shift
            ;;

        -a|--attach)
            check_docker_access

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

# DOOM
Repository for running the experiments on the real robots.

## Requirements 
- docker
  
## Fetch Submodules
```bash
git submodule update --init --recursive
```

## Building the Docker Image
To build the Docker image, use the following command:

```bash
docker build -t mujuni-image unitree_mujoco_container/.devcontainer/.
```

Source the following file to save some aliases:
```bash
source .env.base
```

# Running the container
```bash
dungeon
```

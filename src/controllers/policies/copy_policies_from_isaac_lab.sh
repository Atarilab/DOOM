#!/bin/bash

# Source and destination directories
# NOTE I heavily recommend using absolute paths here. Otherwise you might accidentely have the wrong paths
SRC_DIR="/home/admin_07/project_repos/isaac_lab/IsaacLab/logs/rsl_rl"
DST_DIR="/home/admin_07/DOOM/src/controllers/policies/locomotion_go2"

find "$SRC_DIR" -type f -name "policy.pt" -exec cp --parents {} "$DST_DIR" \;
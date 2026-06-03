#!/usr/bin/env bash
# Run inside the Docker container to rebuild master_manager after plan/code changes.
# Usage:  bash src/plans/rebuild.sh
set -e
cd /home/atari/workspace/DOOM
colcon build --symlink-install --packages-select master_manager
source install/setup.bash
echo "Done. master_manager rebuilt — --plan arg and any other changes are now live."

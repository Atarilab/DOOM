#!/usr/bin/env bash
# Script 1 — Launch the MuJoCo physics simulation.
# Run this FIRST in its own terminal inside Docker.
# Usage (inside Docker):
#   bash src/plans/run_physics.sh          → default Go2 contact sim
#   bash src/plans/run_physics.sh --log myexp  → custom log name
set -e
echo "Starting MuJoCo physics simulation (rl-contact-sim-go2)..."
python src/simulate.py --task rl-contact-sim-go2 "$@"

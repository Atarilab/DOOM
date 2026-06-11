#!/usr/bin/env bash
# Script 3 — Launch RViz for real-time visualization of foot targets, contact plan and goal.
# Run this in a THIRD terminal inside Docker (after run_physics.sh and run_sim.sh).
# Usage (inside Docker):
#   bash src/plans/run_viz.sh
set -e
RVIZ_CFG="src/robots/go2/go2_description/config/go2_rviz.rviz"
echo "Launching RViz with Go2 config..."
echo "Topics to add manually if not shown:"
echo "  /contact_locations   (MarkerArray) — current foot contact locations"
echo "  /feet_trajectories   (MarkerArray) — planned future foot targets"
echo "  /pcbo_goal           (Marker)      — goal position (yellow cylinder)"
ros2 run rviz2 rviz2 -d "$RVIZ_CFG"

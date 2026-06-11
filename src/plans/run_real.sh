#!/usr/bin/env bash
# Shorthand launcher for the real Go2 robot with a PCBO contact plan.
# Default: stops and holds at the knot closest to the goal (--stop-at-goal ON).
# Usage (inside Docker, robot connected):
#   bash src/plans/run_real.sh              → rank-0, stop at goal knot (default)
#   bash src/plans/run_real.sh 2            → rank-2, stop at goal knot
#   bash src/plans/run_real.sh 0 --all-knots  → execute all 6 knots (no stop)
#   bash src/plans/run_real.sh diverse0     → diverse0 (strong-left-lead)
#
# Workflow:
#   1. bash src/plans/rebuild.sh           (once after any code change)
#   2. python src/simulate.py --task rl-contact-sim-go2   (sim terminal)
#   3. bash src/plans/run_sim.sh 0         (verify in simulation first)
#   4. bash src/plans/run_real.sh 0        (deploy on real robot)
#
# Full plan names in src/plans/:
#   rank0 .. rank4
#   diverse0_strong-left-lead  diverse1_front-heavy  diverse2_symmetric-moderate
#   diverse3_symmetric-best    diverse4_right-lateral

PLAN_ARG="${1:-0}"
STOP_AT_GOAL="--stop-at-goal"          # ON by default (safest for real robot)
if [ "${2:-}" = "--all-knots" ]; then STOP_AT_GOAL=""; fi
PLANS_DIR="src/plans"

case "$PLAN_ARG" in
  0|1|2|3|4)
    PLAN_FILE="$PLANS_DIR/go2_plan_pcbo_rank${PLAN_ARG}.json" ;;
  diverse0*)
    PLAN_FILE="$PLANS_DIR/go2_plan_pcbo_diverse0_strong-left-lead.json" ;;
  diverse1*)
    PLAN_FILE="$PLANS_DIR/go2_plan_pcbo_diverse1_front-heavy.json" ;;
  diverse2*)
    PLAN_FILE="$PLANS_DIR/go2_plan_pcbo_diverse2_symmetric-moderate.json" ;;
  diverse3*)
    PLAN_FILE="$PLANS_DIR/go2_plan_pcbo_diverse3_symmetric-best.json" ;;
  diverse4*)
    PLAN_FILE="$PLANS_DIR/go2_plan_pcbo_diverse4_right-lateral.json" ;;
  *)
    PLAN_FILE="$PLAN_ARG" ;;
esac

echo "⚠️  REAL ROBOT — plan: $PLAN_FILE  ${STOP_AT_GOAL}"
echo "Press Ctrl-C within 3s to abort..."
sleep 3

ros2 run master_manager master_node \
  --task rl-contact-real-go2 \
  --plan "$PLAN_FILE" \
  $STOP_AT_GOAL

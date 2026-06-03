#!/usr/bin/env bash
# Shorthand launcher for the Go2 contact simulation with a PCBO plan.
# Usage (inside Docker):
#   bash src/plans/run_sim.sh              → rank-0 (all knots, default)
#   bash src/plans/run_sim.sh 2            → rank-2
#   bash src/plans/run_sim.sh 0 --stop-at-goal  → stop and hold at goal-closest knot
#   bash src/plans/run_sim.sh diverse0     → diverse0 (strong-left-lead)
#   bash src/plans/run_sim.sh diverse4     → diverse4 (right-lateral)
#
# Full plan names in src/plans/:
#   rank0 .. rank4
#   diverse0_strong-left-lead  diverse1_front-heavy  diverse2_symmetric-moderate
#   diverse3_symmetric-best    diverse4_right-lateral

PLAN_ARG="${1:-0}"
STOP_AT_GOAL=""
if [ "${2:-}" = "--stop-at-goal" ]; then STOP_AT_GOAL="--stop-at-goal"; fi
PLANS_DIR="src/plans"

# Accept short aliases: "0".."4" → rank files; "diverse0".."diverse4" → diverse files
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
    PLAN_FILE="$PLAN_ARG" ;;   # treat as a full path
esac

echo "Using plan: $PLAN_FILE  ${STOP_AT_GOAL}"
ros2 run master_manager master_node \
  --task rl-contact-sim-go2 \
  --ui \
  --plan "$PLAN_FILE" \
  $STOP_AT_GOAL

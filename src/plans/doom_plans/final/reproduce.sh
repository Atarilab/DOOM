#!/bin/bash
# Exported: 2026-06-10 19:25  git: 1f6f800
# Run dir:  tmp/experiments/go2_contact_20260610_165549_s0
cd "$(git rev-parse --show-toplevel)"
bash scripts_pcbo/run_go2_contact.sh 0 --config configs_pcbo/go2_contact_gmm_obstacle_lr_5090_sigma_scheduling.json --seed 0

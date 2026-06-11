#!/bin/bash
# Exported: 2026-06-11 05:30  git: 1f6f800
# Run dir:  tmp/experiments/go2_contact_20260611_043507_s0
cd "$(git rev-parse --show-toplevel)"
bash scripts_pcbo/run_go2_contact.sh 0 --config configs_pcbo/go2_contact_gmm_obstacle_lr_5090_lambda1p5_noise0p6.json --seed 0

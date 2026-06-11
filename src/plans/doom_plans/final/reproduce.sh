#!/bin/bash
# Exported: 2026-06-11 00:51  git: 1f6f800
# Run dir:  tmp/experiments/go2_contact_20260610_234557_s0
cd "$(git rev-parse --show-toplevel)"
bash scripts_pcbo/run_go2_contact.sh 0 --config configs_pcbo/go2_contact_gmm_obstacle_lr_5090_const_lambda_1p5.json --seed 0

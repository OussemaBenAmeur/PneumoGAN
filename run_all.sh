#!/usr/bin/env bash
# Reproduce every experiment in docs/2_project_report.md.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

# Experiment 1: full training set (natural imbalance, ~1:2.9)
$PY src/train_gan.py --out results/gan_full
$PY src/evaluate.py  --gan results/gan_full --out results/eval_full

# Experiment 2: severe scarcity. Keep only 10% of NORMAL images (~1:29)
$PY src/train_gan.py --out results/gan_normal10 --minority-fraction 0.1 --diffaug
$PY src/evaluate.py  --gan results/gan_normal10 --out results/eval_normal10

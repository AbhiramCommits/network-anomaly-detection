#!/usr/bin/env bash
set -euo pipefail

# ── Generate synthetic data if not present ──
if [ ! -f data/cicids_sample.csv ]; then
    echo "==> Generating synthetic dataset ..."
    python data/download_data.py
fi

# ── Run full pipeline ──
echo "==> Running anomaly detection pipeline (both models) ..."
python main.py \
    --model both \
    --data data/cicids_sample.csv \
    --output outputs/metrics.json

echo ""
echo "==> Done.  Metrics: outputs/metrics.json"
echo "    Plots:   outputs/figures/"
echo "    Models:  outputs/isolation_forest.pkl"
echo "             outputs/autoencoder.pth"

#!/usr/bin/env bash
# Shard the evaluation across both GPUs.  Set C (the phase grid) is evaluated
# without the channel battery, which is only meaningful where a social channel
# exists to manipulate.
set -u
cd "$(dirname "$0")/.."
mkdir -p results/metrics logs/analysis
N=${N:-6}
for i in $(seq 0 $((N-1))); do
  GPU=$((i % 2))
  CUDA_VISIBLE_DEVICES=$GPU nohup python3 scripts/analyze.py \
      --glob "results/runs/*.pt" --out results/metrics \
      --batch "${BATCH:-192}" --steps "${STEPS:-448}" \
      --shard "$i" --nshard "$N" \
      > "logs/analysis/shard$i.log" 2>&1 &
done
echo "launched $N analysis shards"

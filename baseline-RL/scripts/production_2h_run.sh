#!/bin/bash

# Fixed Production 24-hour Pseudo-RL Training Run
# Uses corrected orchestrator with single training phase per iteration

echo "======================================================================"
echo "FIXED PRODUCTION 24-HOUR PSEUDO-RL TRAINING"
echo "======================================================================"
echo "Start time: $(date)"
echo ""

# Configuration for 24-hour run
# Each iteration: ~1 hour (100 epochs training + rollout)
# 24 hours = 24 iterations

python -m src.orchestrate_training \
    --dataset ./intermediate_data/prepared_dataset_arc_2024.pth \
    ./intermediate_data/prepared_dataset_arc_2025.pth \
    ./intermediate_data/prepared_dataset_barc.pth \
    ./intermediate_data/prepared_dataset_rearc.pth \
    --iterations 2 \
    --initial-epochs 10 \
    --mixed-epochs 10 \
    --batch-size 2 \
    --samples-per-epoch 30 \
    --embed-size 768 \
    --num-layers 8 \
    --heads 8 \
    --num-kv-heads 2 \
    --learning-rate 0.0002 \
    --warmup-epochs 1 \
    --trajectory-samples 10 \
    --temperature 0.8 \
    --accumulation-steps 4 \
    --max-seq-length 2700 \
    --yes

echo ""
echo "======================================================================"
echo "Training complete!"
echo "End time: $(date)"
echo "======================================================================"
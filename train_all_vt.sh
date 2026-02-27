#!/bin/bash
# VTDexManip VT-JointPretrain (vt_all_cls) Sequential Training Script
# Run from: pi-touch/libs/VTDexManip/
#
# Usage:
#   nohup bash train_all_vt.sh > train_all_vt.log 2>&1 &
#   tail -f train_all_vt.log
#
# All tasks: max_iterations=3000, ~10h each, Total ~50h
#

GPU="cuda:0"
SEED=111
MAX_ITER=3000

TASKS=(
    "bottle_cap"
    "slide"
    "reorient_down"
    "reorient_up"
    "handover"
)

echo "=========================================="
echo " VT-JointPretrain Training — $(date)"
echo " GPU: ${GPU}, Seed: ${SEED}, MaxIter: ${MAX_ITER}"
echo " Tasks: ${TASKS[*]}"
echo "=========================================="

# Override max_iterations in all PPO configs
for TASK in "${TASKS[@]}"; do
    CFG="config/algos/ppo/${TASK}.yaml"
    sed -i "s/max_iterations:.*/max_iterations: ${MAX_ITER}/" "${CFG}"
    echo " [Config] ${CFG} -> max_iterations: ${MAX_ITER}"
done

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "=========================================="
    echo " START: ${TASK}-vt_all_cls — $(date)"
    echo "=========================================="

    python train_agent.py \
        --task ${TASK}-vt_all_cls \
        --rl_device ${GPU} \
        --seed ${SEED} \
        --headless

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[DONE] ${TASK}-vt_all_cls finished successfully — $(date)"
    else
        echo "[FAIL] ${TASK}-vt_all_cls exited with code ${EXIT_CODE} — $(date)"
    fi
done

echo ""
echo "=========================================="
echo " ALL TASKS COMPLETE — $(date)"
echo "=========================================="

#!/bin/bash

TASK="TestPickItUp"
NUM_ENVS=20
VIDEO_LENGTH=1500

CHECKPOINTS=(
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/test_pick_it_up/2026-03-25_05-38-10_Run-0tueilsy-st-13/model_2999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/test_pick_it_up/2026-03-25_04-54-55_Run-0tueilsy-st-13/model_2999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/test_pick_it_up/2026-03-25_03-26-51_Run-0tueilsy-st-13/model_2999.pt"
"/home/shaotongchen/workspace_eureka/IsaacLabEureka/logs/rl_runs/rsl_rl_eureka/test_pick_it_up/2026-03-25_00-34-28_Run-0tueilsy-st-13/model_2999.pt"
)

# ---------- PLAY LOOP ----------
for CKPT in "${CHECKPOINTS[@]}"; do
    echo "===== Playing checkpoint: $CKPT ====="

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
        --task="$TASK" \
        --num_envs=$NUM_ENVS \
        --video \
        --video_length=$VIDEO_LENGTH \
        --checkpoint="$CKPT"

    echo "===== Finished: $CKPT ====="
done

echo "===== All runs finished ====="
#!/usr/bin/env bash
# Arch F (Arch E without differential attention) training: start, stop and resume on 1 or 2 GPUs.
#
#   bash phase1/launch_archF.sh gpu1     # train on GPU 1 only (resumes from ckpt_latest.pt if it exists)
#   bash phase1/launch_archF.sh both     # train on GPUs 0 and 1 with DDP (resumes likewise)
#   bash phase1/launch_archF.sh stop     # save ckpt_latest.pt after the current step and exit
#   bash phase1/launch_archF.sh status   # running processes + last log lines
#
# Switching GPUs: `stop`, wait until it reports stopped, then `both` (or `gpu1`). Each step's data
# depends only on the step number, so the run continues exactly where it stopped whatever the GPU count.
set -u
cd "$(dirname "$0")/.."
RUN=runs/archF_nodiff_145M
CFG=phase1/configs/archF.json
BIN="$(dirname "$(command -v python)")"
mkdir -p "$RUN"

running() { pgrep -f "phase1/train.py --config $CFG" > /dev/null; }

case "${1:-}" in
  gpu1|both)
    if running; then echo "Arch F is already running; use 'stop' first"; exit 1; fi
    if [ "$1" = gpu1 ]; then GPUS=1; N=1; else GPUS=0,1; N=2; fi
    [ -f "$RUN/ckpt_latest.pt" ] && echo "resuming from $RUN/ckpt_latest.pt" || echo "starting from scratch"
    echo "=== $(date '+%F %T') launch on GPU(s) $GPUS ===" >> "$RUN/train.log"
    CUDA_VISIBLE_DEVICES=$GPUS setsid nohup "$BIN/torchrun" --standalone --nproc_per_node $N \
      phase1/train.py --config $CFG --resume >> "$RUN/train.log" 2>&1 < /dev/null &
    echo "launched on GPU(s) $GPUS; log: $RUN/train.log"
    ;;
  stop)
    if ! running; then echo "Arch F is not running"; exit 0; fi
    touch "$RUN/STOP"
    echo "STOP requested; waiting for the current step to finish and ckpt_latest.pt to be saved..."
    while running; do sleep 5; done
    grep "STOP file found" "$RUN/train.log" | tail -1
    echo "stopped"
    ;;
  status)
    running && echo "running" || echo "not running"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
    grep -E "^step|val_bm|STOP|launch|Error" "$RUN/train.log" | tail -4
    ;;
  *)
    sed -n 2,10p "$0"; exit 1 ;;
esac

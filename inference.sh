#!/bin/bash
# Start the script in its own process group
set -m

ARGS="--data_path /workspace-SR004.nfs2/data/long_vqa_synth/ --exp_name main --semaphore_limit 16 --batch_size 300"
TEXT1_ARGS="--text_json_path /workspace-SR004.nfs2/data/long_vqa_synth/main_1mv/all_text_serialized_questions.json --exp_name main_1mv  --semaphore_limit 32 --batch_size 300"
TEXT_ARGS="--text_json_path /workspace-SR004.nfs2/data/long_vqa_synth/main/all_text_serialized_questions.json  --exp_name main --semaphore_limit 32 --batch_size 300"
(
    python scripts/openai_server_inference.py --port 8003 $TEXT1_ARGS &
    python scripts/openai_server_inference.py --port 8003 $TEXT_ARGS &
    # python scripts/openai_server_inference.py --port 8002 $TEXT_ARGS &
    # python scripts/openai_server_inference.py --port 8003 $TEXT_ARGS &
    # python scripts/openai_server_inference.py --port 8014 $ARGS &
    # python scripts/openai_server_inference.py --port 8013 $ARGS &
    # python scripts/openai_server_inference.py --port 8011 $ARGS &
    # python scripts/openai_server_inference.py --port 8011 $ARGS &
    # python scripts/openai_server_inference.py --port 8016 $ARGS &
    # python scripts/openai_server_inference.py --port 8024 $ARGS &
    # python scripts/openai_server_inference.py --port 8025 $ARGS &
    wait
) &

# Capture the Process Group ID (PGID) of the subshell
PGID=$!

# Trap EXIT and clean up all child processes in the group
trap "echo 'Killing process group $PGID'; kill -9 -- -$PGID" EXIT
wait

echo "Inference complete."
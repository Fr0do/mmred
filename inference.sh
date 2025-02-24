#!/bin/bash
# Start the script in its own process group
set -m

ARGS="--data_path /home/jovyan/shares/SR004.nfs2/data/long_vqa_synth/main_1mv"

(
    python scripts/openai_server_inference.py --port 8023 $ARGS &
    python scripts/openai_server_inference.py --port 8025 $ARGS &
    python scripts/openai_server_inference.py --port 8020 --text_json_path /home/jovyan/shares/SR004.nfs2/abdullaeva/video-llm/Video-bAbI/long-vqa-v1/data/all_text_serialized_questions.json &
    python scripts/openai_server_inference.py --port 8007 $ARGS &
    # python scripts/openai_server_inference.py --port 8004 $ARGS &
    # python scripts/openai_server_inference.py --port 8005 $ARGS &
    # python scripts/openai_server_inference.py --port 8006 &
    # python scripts/openai_server_inference.py --port 8007 &
    # python scripts/openai_server_inference.py --port 8008 &
    # python scripts/openai_server_inference.py --port 8009 &
    wait
) &

# Capture the Process Group ID (PGID) of the subshell
PGID=$!

# Trap EXIT and clean up all child processes in the group
trap "echo 'Killing process group $PGID'; kill -9 -- -$PGID" EXIT
wait

echo "Inference complete."
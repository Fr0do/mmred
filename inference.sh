#!/bin/bash
# Start the script in its own process group
set -m

ARGS="--data_path /workspace-SR004.nfs2/data/long_vqa_synth/ --exp_name main_1mv --semaphore_limit 12 --batch_size 32"
TEXT1_ARGS="--text_json_path /workspace-SR004.nfs2/data/long_vqa_synth/main_1mv/all_text_serialized_questions.json --exp_name main_1mv  --semaphore_limit 48 --batch_size 50"
TEXT3_ARGS="--text_json_path /workspace-SR004.nfs2/data/long_vqa_synth/main_1mv/all_text_serialized_questions_mask_0.250.json --exp_name main_1mv_4th_mask  --semaphore_limit 48 --batch_size 50"
TEXT4_ARGS="--text_json_path /workspace-SR004.nfs2/data/long_vqa_synth/main_1mv/all_text_serialized_questions_error_0.050.json --exp_name main_1mv_error_050  --semaphore_limit 48 --batch_size 50"
TEXT_ARGS="--text_json_path /workspace-SR004.nfs2/data/long_vqa_synth/main/all_text_serialized_questions.json  --exp_name main_1mv --semaphore_limit 128 --batch_size 25"
TEXT_NLG_ARGS="--text_json_path /workspace-SR004.nfs2/data/long_vqa_synth/main_1mv/text_description_serialized.json  --exp_name nlg_1mv --semaphore_limit 128 --batch_size 300"
(
    python scripts/openai_server_inference.py --port 8003 $TEXT_ARGS --prefix_question --thinking &
    # python scripts/openai_server_inference.py --port 8003 $TEXT_ARGS --prefix_question &
    wait
) &

# Capture th_errore Process Group ID (PGID) of th_errore subshell
PGID=$!

# Trap EXIT and clean up all child processes in th_errore group
trap "echo 'Killing process group $PGID'; kill -9 -- -$PGID" EXIT
wait

echo "Inference complete."